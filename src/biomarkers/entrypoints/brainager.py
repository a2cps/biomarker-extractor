import gzip
import logging
import shutil
import typing
from pathlib import Path

from biomarkers import task_runners
from biomarkers.cli import dask_config
from biomarkers.entrypoints import tapis
from biomarkers.flows import brainager


class BrainagerEntrypoint(tapis.TapisEntrypoint):
    n_workers: int = 1

    def run_flow(self) -> list[Path]:
        dask_config.set_config()

        staged_ins, staged_outs = self._stage(self.ins, self.outs)

        brainager.brainager_flow.with_options(
            task_runner=task_runners.DaskTaskRunner(
                cluster_kwargs={
                    "n_workers": self.n_workers,
                    "threads_per_worker": 1,
                    "dashboard_address": None,
                }
            )
        )(images=staged_ins, outdirs=staged_outs, return_state=True)

        return staged_outs

    def _stage(
        self,
        ins: typing.Iterable[Path],
        outs: typing.Iterable[Path],
    ) -> tuple[list[Path], list[Path]]:
        niis = []
        output_dirs = []

        # dirs may point to broken symlinks, or folders might not exist
        # so, need to ensure that we get one output dir for each input dir
        for ind, outd in zip(ins, outs, strict=True):
            logging.info(f"Looking for *T1w.nii.gz in {ind}")
            nii = list(d for d in ind.rglob("*T1w.nii.gz"))
            if not len(nii) == 1:
                logging.warning(
                    f"Unexpected number of *T1w.nii.gz found within {ind}, {len(nii)=}"
                )
            else:
                logging.info(f"Staging {nii[0]}")
                with gzip.open(nii[0], "rb") as f_in:
                    with open(self.stage_dir / nii[0].stem, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                        niis.append(Path(f_out.name))
                output_dirs.append(self.stage_dir / outd)
        return niis, output_dirs
