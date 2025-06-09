import logging
import shutil
import typing
from pathlib import Path

from biomarkers.entrypoints import tapismpi
from biomarkers.flows import postdtifit


class PostDTIFitEntrypoint(tapismpi.TapisMPIEntrypoint):
    participant_label: typing.Sequence[str]
    ses_label: typing.Sequence[str]
    qsiprepdir: typing.Sequence[Path]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "dtimetrics").exists()

    def prep(self, tmpd_in: Path) -> Path:
        tmp_qsiprep_dirs: dict[int, Path] = dict()
        for src in tapismpi.iterate_byrank_serial(self.qsiprepdir, self.RANK):
            logging.info(f"Staging files for {src=} -> {tmpd_in=}")
            try:
                tmp_qsiprep_dirs[self.RANK] = shutil.copytree(
                    src, tmpd_in, ignore=self.stage_ignore_patterns, dirs_exist_ok=True
                )
            except Exception:
                logging.error("Failed to stage qsiprep. Subsequent steps will fail.")
        if self.RANK not in tmp_qsiprep_dirs:
            logging.error("Never staged qsiprep. Subsequent steps will fail.")
            return tmpd_in

        return Path(tmp_qsiprep_dirs[self.RANK])

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        qsiprepdir = self.prep(tmpd_in=in_dir)
        logging.info(f"extracting gift measures from {in_dir}")
        await postdtifit.postdtifit_flow(
            dtifit=in_dir,
            outdir=out_dir,
            sub=self.participant_label[self.RANK],
            ses=self.ses_label[self.RANK],
            qsiprep=qsiprepdir,
        )
