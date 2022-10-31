from pathlib import Path

import prefect
from prefect_shell import shell_run_command
from prefect_dask import DaskTaskRunner

from ..models.fslanat import FSLAnatResult
from .. import utils


def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return Path(out / basename).with_suffix(".anat").absolute()


@prefect.task
async def _fslanat(image: Path, out: Path) -> Path:
    basename = utils.img_stem(image)
    anat = _predict_fsl_anat_output(out, basename)

    # if the output already exists, we don't want this to run again.
    # fsl_anat automatically and always adds .anat to the value of -o, so we check for the existence
    # of that predicted output, but then feed in the unmodified value of -o to the task.
    if not anat.exists():
        await shell_run_command.fn(command=f"fsl_anat -i {image} -o {out / basename}")

    fslanat = FSLAnatResult.from_root(anat)
    filename = out / f"{basename}_first.tsv"
    fslanat.write_volumes(filename=filename)
    return filename


@prefect.flow
def fslanat_flow(images: frozenset[Path], out: Path) -> None:
    _fslanat.map(images, out=out)
