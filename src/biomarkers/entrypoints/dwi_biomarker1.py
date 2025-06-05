import logging
import shutil
import typing
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.flows import dwi_biomarker1 as dwi_bm1_flow


class DWIBiomarker1Entrypoint(tapismpi.TapisMPIEntrypoint):
    participant_label: typing.Sequence[str]
    ses_label: typing.Sequence[str]
    bedpostxdir: typing.Sequence[Path]
    roi_dir: Path = Path("/opt/tapis/rois")

    def get_args(self, qsiprepdir: Path, outdir: Path, bedpostxdir: Path) -> list[str]:
        return [
            "probtrackx2_voxelwise",
            str(self.participant_label[self.RANK]),
            str(self.ses_label[self.RANK]),
            str(qsiprepdir),
            str(bedpostxdir),
            str(outdir),
            str(self.roi_dir),
        ]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "probtrackx").exists() and (
            len(list((output_dir_to_check).rglob("*tsv"))) == 1
        )

    def prep(self, tmpd_in: Path) -> Path:
        tmp_bedpostx_dirs: dict[int, Path] = dict()
        for src in tapismpi.iterate_byrank_serial(self.bedpostxdir, self.RANK):
            logging.info(f"Staging files for {src=} -> {tmpd_in=}")
            try:
                tmp_bedpostx_dirs[self.RANK] = shutil.copytree(
                    src, tmpd_in, ignore=self.stage_ignore_patterns, dirs_exist_ok=True
                )
            except Exception:
                logging.error("Failed to stage bedpostx. Subsequent steps will fail.")
        if self.RANK not in tmp_bedpostx_dirs:
            logging.error("Never staged bedpostx. Subsequent steps will fail.")
            return tmpd_in

        return Path(tmp_bedpostx_dirs[self.RANK])

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        bedpostxdir = self.prep(in_dir)
        async with utils.subprocess_manager(
            log=out_dir / f"probtrackx2_rank-{self.RANK}.log",
            args=self.get_args(
                qsiprepdir=in_dir, outdir=out_dir, bedpostxdir=bedpostxdir
            ),
        ) as proc:
            await proc.wait()

            if proc.returncode and proc.returncode > 0:
                msg = f"bedpostx failed with {proc.returncode=}"
                raise RuntimeError(msg)

        dwi_bm1_flow.dwi_biomarker1_flow(
            outdir=out_dir,
            sub=self.participant_label[self.RANK],
            ses=self.ses_label[self.RANK],
        )
