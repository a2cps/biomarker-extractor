import logging
import shutil
import typing
from pathlib import Path

from biomarkers import task_runners
from biomarkers.cli import dask_config
from biomarkers.entrypoints import tapis
from biomarkers.flows import fslanat
from biomarkers.models import fslanat as fslanat_models


class FSLAnatEntrypoint(tapis.TapisEntrypoint):
    n_workers: int = 1
    precrop: typing.Sequence[bool] | None = None
    mask_high_voxels: typing.Sequence[bool] | None = None
    strongbias: typing.Sequence[bool] | None = None

    def run_flow(self) -> list[Path]:
        dask_config.set_config()

        staged_ins, staged_outs = self._stage()

        fslanat.fslanat_flow.with_options(
            task_runner=task_runners.DaskTaskRunner(
                cluster_kwargs={
                    "n_workers": self.n_workers,
                    "threads_per_worker": 1,
                    "dashboard_address": None,
                }
            )
        )(
            images=staged_ins,
            outdirs=staged_outs,
            precrops=self.precrop,
            strongbias=self.strongbias,
            mask_high_voxels=self.mask_high_voxels,
            return_state=True,
        )

        return staged_outs

    def _stage(self) -> tuple[list[Path], list[Path]]:
        niis = []
        output_dirs = []

        # dirs may point to broken symlinks, or folders might not exist
        # so, need to ensure that we get one output dir for each input dir
        for ind, outd in zip(self.ins, self.outs, strict=True):
            logging.info(f"Looking for *T1w.nii.gz in {ind}")
            nii = list(d for d in ind.rglob("*T1w.nii.gz"))
            if not len(nii) == 1:
                msg = f"Unexpected number of *T1w.nii.gz found within {ind}, {len(nii)=}"
                raise AssertionError(msg)
            else:
                logging.info(f"Staging {nii[0]}")
                staged_file = shutil.copyfile(
                    nii[0], self.stage_dir / nii[0].name
                )
                niis.append(staged_file)
                output_dirs.append(self.stage_dir / outd)
        return niis, output_dirs

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        out = True
        anatdirs = list(output_dir_to_check.glob("*.anat"))
        if not len(anatdirs) == 1:
            logging.error(
                f"Unexpected number of .anat dirs found in {output_dir_to_check}: {anatdirs=}"
            )
            out = False
        try:
            out &= isinstance(
                fslanat_models.FSLAnatResult.from_root(anatdirs[0]),
                fslanat_models.FSLAnatResult,
            )
        except Exception as e:
            logging.error(e)
            out = False

        return out
