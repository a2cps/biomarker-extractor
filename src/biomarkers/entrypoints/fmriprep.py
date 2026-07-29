import json
import logging
import shutil
import tempfile
import typing
from pathlib import Path

import nibabel as nb

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import fmriprep as fmriprep_models

MIN_FMRI_FRAMES = 200


def extend_arg(
    args: list[str],
    name: str,
    value: str | int | bool | Path | None = None,
):
    if value:
        match name:
            case "--output-spaces":
                for space in str(value).split(" "):
                    args.extend([name, str(space)])
            case _:
                args.extend([name, str(value)])


class FMRIPRepEntrypoint(tapismpi.TapisMPIEntrypoint):
    fs_license_file: Path
    n_workers: int | None = None
    mem_mb: int | None = None
    cifti_output: fmriprep_models.CIFTI_OUTPUT = "91k"
    dummy_scans: int | None = None
    bold2anat_dof: fmriprep_models.BOLD2ANAT_DOF = 6
    output_spaces: typing.Sequence[fmriprep_models.OUTPUT_SPACE] = typing.get_args(
        fmriprep_models.OUTPUT_SPACE
    )
    anat_only: typing.MutableSequence[bool] | None = None
    derivatives: typing.Sequence[Path] | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list((output_dir_to_check / "fmriprep").glob("*html"))) > 0
        )

    def get_args(self, bidsdir: Path, outdir: Path, work_dir: Path) -> list[str]:
        args = [
            "/bin/bash",
            "/shell-hook.sh",
            "/app/.pixi/envs/fmriprep/bin/fmriprep",
            "--notrack",
            "--return-all-components",
            "--ignore",
            "fmap-jacobian",
        ]
        if self.anat_only and self.anat_only[self.RANK]:
            args.append("--anat-only")
        if self.derivatives:
            extend_arg(args, "--derivatives", str(self.derivatives[self.RANK]))

        to_extend = {
            "--fs-license-file": self.fs_license_file,
            "--n-cpus": self.n_workers,
            "--mem_mb": self.mem_mb,
            "--bold2anat-dof": self.bold2anat_dof,
            "--cifti-output": self.cifti_output,
            "--output-spaces": " ".join(self.output_spaces),
            "--dummy-scans": self.dummy_scans,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        args.extend([str(bidsdir), str(outdir / "fmriprep"), "participant"])

        return args

    def prep(self, in_dir: Path) -> None:

        if self.anat_only and self.anat_only[self.RANK]:
            return

        logging.info("Looking for short scans")
        to_delete = []
        for f in in_dir.rglob("*bold.nii.gz"):
            nii = nb.nifti1.Nifti1Image.load(f)
            if nii.shape[-1] < MIN_FMRI_FRAMES:
                to_delete.append(f)
                to_delete.append(Path(str(f).replace("nii.gz", "json")))
                if (
                    events_tsv := Path(str(f).replace("bold.nii.gz", "events.tsv"))
                ).exists():
                    to_delete.append(events_tsv)

        for f in to_delete:
            logging.info(f"removing {f}")
            f.unlink()
            # also need to remove references to this file in fieldmap metadata
            if (fmaps := f.parent.parent / "fmap").exists():
                sesdir = fmaps.parent
                subdir = sesdir.parent
                for fmap in fmaps.glob("*fmrib0*.json"):
                    metadata = json.loads(fmap.read_text())
                    if "IntendedFor" in metadata:
                        intended_for = [
                            str(i)
                            for i in metadata.get("IntendedFor")
                            if i != str(f.relative_to(subdir))
                        ]
                        metadata["IntendedFor"] = intended_for
                    fmap.write_text(json.dumps(metadata))

            # assuming that there is no *scans.tsv file which could also refer to these files

        # if there were bold files but now there aren't, delete the func folder and
        # convert this to an anat-only run
        if len(to_delete) and not len(list(in_dir.rglob("*bold.nii.gz"))):
            to_delete[0].parent.rmdir()

            if self.anat_only:
                self.anat_only[self.RANK] = True

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:

        self.prep(in_dir)

        with tempfile.TemporaryDirectory() as tmpd:
            async with utils.subprocess_manager(
                log=out_dir / f"fmriprep_rank-{self.RANK}.log",
                args=self.get_args(bidsdir=in_dir, outdir=out_dir, work_dir=Path(tmpd)),
            ) as proc:
                await proc.wait()
                if proc.returncode and proc.returncode > 0:
                    # remove folder so that archiving detects that there was a failure
                    # and sends logs to failure_dst_dir
                    if (outdir_fmriprep := out_dir / "fmriprep").exists():
                        shutil.rmtree(outdir_fmriprep)
                    msg = f"fmriprep failed with {proc.returncode=}"
                    raise RuntimeError(msg)
