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
    n_workers: int = 1

    def get_args(self, qsiprepdir: Path, outdir: Path) -> list[str]:
        return [
            "probtrackx2_voxelwise",
            str(self.participant_label[self.RANK]),
            str(self.ses_label[self.RANK]),
            str(qsiprepdir),
            str(self.bedpostxdir[self.RANK]),
            str(outdir),
            str(self.roi_dir),
            str(self.n_workers),
        ]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "probtrackx").exists()

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        async with utils.subprocess_manager(
            log=tmpd_out / f"probtrackx2_rank-{self.RANK}.log",
            args=self.get_args(qsiprepdir=tmpd_in, outdir=tmpd_out),
        ) as proc:
            await proc.wait()

            if proc.returncode and proc.returncode > 0:
                msg = f"bedpostx failed with {proc.returncode=}"
                raise RuntimeError(msg)

            dwi_bm1_flow.dwi_biomarker1_flow(
                outdir=tmpd_out,
                participant_label=self.participant_label[self.RANK],
                session_label=self.ses_label[self.RANK],
            )
