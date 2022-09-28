from __future__ import annotations
from pathlib import Path

import pydantic

import prefect
from prefect.tasks import task_input_hash
from prefect_shell import shell_run_command
from prefect_dask import DaskTaskRunner

from ..models.fslanat import FSLAnatResult
from .. import utils


@prefect.task
def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return Path(out / basename).with_suffix(".anat").absolute()


@prefect.task
def build_fslanat(root: Path) -> FSLAnatResult:
    """Run fsl_anat on an image and return the results."""
    try:
        out = FSLAnatResult.from_root(root)
        return out
    except pydantic.ValidationError as e:
        raise ValueError(f"Could not parse fsl_anat results: {e}")


@prefect.task(cache_key_fn=task_input_hash)
def write_first_volumes(fslanat: FSLAnatResult, filename: Path) -> None:
    fslanat.write_volumes(filename=filename)


@prefect.task
def get_cmd(anat: Path, image: Path, out: Path, basename: str) -> str:
    # if the output already exists, we don't want this to run again.
    # if it exists, we change the command to just "echo"
    # fsl_anat automatically and always adds .anat to the value of -o, so we check for the existence
    # of that predicted output, but then feed in the unmodified value of -o to the task.
    if anat.exists():
        cmd = f"echo 'Found existing FSLAnatResult at output location, {anat}. Assuming complete, so will skip (remove {anat} to run).'"
    else:
        cmd = f"fsl_anat -i {image} -o {out / basename}"
    return cmd


@prefect.flow(task_runner=DaskTaskRunner)
def fslanat_flow(images: set[Path], out: Path) -> None:
    for image in images:
        basename = utils.img_stem(image)
        anat = _predict_fsl_anat_output.submit(out, basename)
        cmd = get_cmd.submit(anat=anat, image=image, out=out, basename=basename)

        last_line = shell_run_command.submit(command=cmd)
        fslanat = build_fslanat.submit(root=anat, wait_for=[last_line])
        write_first_volumes.submit(fslanat, out / f"{basename}_first.tsv")
