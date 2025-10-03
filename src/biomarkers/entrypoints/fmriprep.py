import shutil
import tempfile
import typing
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import fmriprep as fmriprep_models


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
    cifti_output: fmriprep_models.CIFTI_OUTPUT | None = None
    dummy_scans: int | None = None
    bold2anat_dof: fmriprep_models.BOLD2ANAT_DOF = 6
    output_spaces: typing.Sequence[fmriprep_models.OUTPUT_SPACE] = typing.get_args(
        fmriprep_models.OUTPUT_SPACE
    )
    anat_only: typing.Sequence[bool] | None = None

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
            "--dummy-scans": self.dummy_scans,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        args.extend([str(bidsdir), str(outdir / "fmriprep"), "participant"])

        return args

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
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
