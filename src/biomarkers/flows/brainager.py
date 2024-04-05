import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nb
import prefect
import pydantic
from nibabel import affines

from biomarkers.models.brainager import BrainAgeResult

# https://github.com/nipy/nitransforms/blob/0027d1b0ea752d5aa88558678d2b27510366314b/nitransforms/io/afni.py#L18
OBLIQUITY_THRESHOLD_DEG = 0.01


def _is_oblique(image: Path) -> bool:
    affine = nb.nifti1.load(image).affine
    if affine is None:
        msg = f"Input {image=} has no affine?"
        raise AssertionError(msg)

    # https://github.com/nipy/nitransforms/blob/0027d1b0ea752d5aa88558678d2b27510366314b/nitransforms/io/afni.py#L213-L229
    return (
        affines.obliquity(affine).max() * 180 / math.pi
    ) > OBLIQUITY_THRESHOLD_DEG


def _deoblique(image: Path, dst: Path) -> Path:
    try:
        completed_process = subprocess.run(
            [
                "3dWarp",
                "-prefix",
                f"{dst.name.removesuffix('.gz')}",
                "-cubic",
                "-deoblique",
                f"{image}",
            ],
            capture_output=True,
        )
    except BaseException as e:
        msg = (
            f"Failed during deoblique of {image=}. Is AFNI's 3dWarp available?"
        )
        logging.error(msg)
        raise e

    (dst.parent / "obliquity.mat").write_bytes(completed_process.stdout)

    return dst


@prefect.task
def _brainager(image: Path, out: Path):

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        nii = tmpdir / image.name
        if _is_oblique(image):
            logging.warning(f"Detected oblique {image=}")
            _deoblique(image, nii)
        else:
            shutil.copy2(image, nii)

        completed_process = subprocess.run(
            [  # noqa: S603
                "brainageR",
                "-f",
                f"{nii}",
                "-o",
                f"{nii.with_suffix('.csv')}",
            ],
            capture_output=True,
            cwd=tmpdir,
        )
        # test that all expected outputs exist
        try:
            BrainAgeResult.from_nii(nii=nii)
        except pydantic.ValidationError as e:
            # if not, raise error and show logs
            msg = f"""
            stdout:
            {str(completed_process.stdout)}

            stderr:
            {str(completed_process.stderr)}
            """
            logging.error(msg=msg)
            raise e

        # add logs
        Path(tmpdir / "out.txt").write_bytes(completed_process.stdout)
        Path(tmpdir / "err.txt").write_bytes(completed_process.stderr)

        # store everything in final destination
        shutil.copytree(tmpdir, out, dirs_exist_ok=True)


@prefect.flow
def brainager_flow(images: list[Path], outdirs: list[Path]) -> None:

    for image, out in zip(images, outdirs, strict=True):
        _brainager.submit(image, out=out)
