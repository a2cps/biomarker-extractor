import logging
import shutil
import tempfile
import typing
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import fmriprep as fmriprep_models


def get_synthstrip_args(
    src: Path,
    model: Path,
    mask: Path,
    n_workers: int | None = None,
    no_csf: bool = False,
) -> list[str]:
    return (
        [
            "synthstrip",
            "-i",
            str(src),
            "-m",
            str(mask),
            "-n",
            str(n_workers if n_workers else 1),
            "--model",
            str(model),
        ]
        + ["--no-csf"]
        if no_csf
        else []
    )


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
    synthstrip_model: Path
    n_workers: int | None = None
    mem_mb: int | None = None
    cifti_output: fmriprep_models.CIFTI_OUTPUT = "91k"
    dummy_scans: int | None = None
    bold2anat_dof: fmriprep_models.BOLD2ANAT_DOF = 6
    output_spaces: typing.Sequence[fmriprep_models.OUTPUT_SPACE] = typing.get_args(
        fmriprep_models.OUTPUT_SPACE
    )
    anat_only: typing.Sequence[bool] | None = None
    no_csf: bool = False

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list((output_dir_to_check / "fmriprep").glob("*html"))) > 0
        )

    def get_args(self, bidsdir: Path, outdir: Path, work_dir: Path) -> list[str]:
        args = ["fmriprep", "--notrack", "--return-all-components"]
        if self.anat_only and self.anat_only[self.RANK]:
            args.append("--anat-only")
        to_extend = {
            "--fs-license-file": self.fs_license_file,
            "--n-cpus": self.n_workers,
            "--mem_mb": self.mem_mb,
            "--bold2anat-dof": self.bold2anat_dof,
            "--cifti-output": self.cifti_output,
            "--output-spaces": " ".join(self.output_spaces),
            "--derivatives": f"synthstrip={outdir}/synthstrip",
            "--dummy-scans": self.dummy_scans,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        args.extend([str(bidsdir), str(outdir / "fmriprep"), "participant"])

        return args

    async def prep(self, tmpd_in: Path, tmpd_out: Path) -> Path | None:
        logging.info("Generating brainmask with synthstrip")
        maybe_nii = list(d for d in tmpd_in.rglob("*T1w.nii.gz"))
        if len(maybe_nii) == 0:
            msg = f"Did not find any *T1w.nii.gz in {tmpd_in}"
            raise AssertionError(msg)
        elif len(maybe_nii) > 1:
            logging.warning(
                f"Unexpected number of  *T1w.nii.gz found in {tmpd_in}: {maybe_nii}. Taking first."
            )
        nii = maybe_nii[0]
        sub = utils.get_sub(nii)
        ses = utils.get_ses(nii)
        mask = (
            tmpd_out
            / "synthstrip"
            / f"sub-{sub}"
            / f"ses-{ses}"
            / "anat"
            / nii.name.replace("T1w.nii.gz", "desc-brain_mask.nii.gz")
        )
        mask.parent.mkdir(parents=True)

        async with utils.subprocess_manager(
            log=tmpd_out / f"synthstrip_rank-{self.RANK}.log",
            args=get_synthstrip_args(
                nii,
                model=self.synthstrip_model,
                mask=mask,
                n_workers=self.n_workers,
                no_csf=self.no_csf,
            ),
        ) as proc:
            await proc.wait()
            if proc.returncode and proc.returncode > 0:
                msg = f"synthstrip failed with {proc.returncode=}"
                raise RuntimeError(msg)

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        await self.prep(tmpd_in, tmpd_out)
        with tempfile.TemporaryDirectory() as tmpd:
            async with utils.subprocess_manager(
                log=tmpd_out / f"fmriprep_rank-{self.RANK}.log",
                args=self.get_args(
                    bidsdir=tmpd_in, outdir=tmpd_out, work_dir=Path(tmpd)
                ),
            ) as proc:
                await proc.wait()
                if proc.returncode and proc.returncode > 0:
                    # remove folder so that archiving detects that there was a failure
                    # and sends logs to failure_dst_dir
                    if (outdir_fmriprep := tmpd_out / "fmriprep").exists():
                        shutil.rmtree(outdir_fmriprep)
                    msg = f"fmriprep failed with {proc.returncode=}"
                    raise RuntimeError(msg)
