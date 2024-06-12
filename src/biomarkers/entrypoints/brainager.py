import gzip
import logging
import math
import shutil
import tempfile
from pathlib import Path

import nibabel as nb
from nibabel import affines

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import brainager as brainager_models

# https://github.com/nipy/nitransforms/blob/0027d1b0ea752d5aa88558678d2b27510366314b/nitransforms/io/afni.py#L18
OBLIQUITY_THRESHOLD_DEG = 0.01


def is_oblique(image: Path) -> bool:
    affine = nb.nifti1.load(image).affine
    if affine is None:
        msg = f"Input {image=} has no affine?"
        raise AssertionError(msg)

    # https://github.com/nipy/nitransforms/blob/0027d1b0ea752d5aa88558678d2b27510366314b/nitransforms/io/afni.py#L213-L229
    return (
        affines.obliquity(affine).max() * 180 / math.pi
    ) > OBLIQUITY_THRESHOLD_DEG


def get_brainager_args(nii: Path) -> list[str]:
    return ["brainageR", "-f", str(nii), "-o", str(nii.with_suffix(".csv"))]


class BrainagerEntrypoint(tapismpi.TapisMPIEntrypoint):
    n_workers: int = 1

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        out = True
        t1 = list(output_dir_to_check.glob("sub*T1w.nii"))
        if not len(t1) == 1:
            logging.error(
                f"Unexpected number of niftis found in {output_dir_to_check}: {t1=}"
            )
            out = False
        try:
            out &= isinstance(
                brainager_models.BrainAgeResult.from_nii(t1[0]),
                brainager_models.BrainAgeResult,
            )
        except Exception as e:
            logging.error(e)
            out = False

        return out

    async def deoblique(self, image: Path) -> int | None:
        with tempfile.TemporaryDirectory() as tmpd:
            deobliqued = Path(tmpd) / "deobliqued.nii"
            async with utils.subprocess_manager(
                args=[
                    "3dWarp",
                    "-prefix",
                    str(deobliqued),
                    "-cubic",
                    "-deoblique",
                    str(image),
                ],
                log=image.parent / f"3dWarp_rank-{self.RANK}.log",
            ) as proc:
                await proc.wait()
            image.unlink()
            shutil.move(deobliqued, image, copy_function=shutil.copyfile)
        return proc.returncode

    async def prep(self, tmpd_in: Path, tmpd_out: Path) -> Path | None:

        logging.info(f"Looking for *T1w.nii.gz in {tmpd_in}")
        maybe_nii = list(d for d in tmpd_in.rglob("*T1w.nii.gz"))
        if len(maybe_nii) == 0:
            logging.error(f"Did not find any *T1w.nii.gz in {tmpd_in}")
            return
        elif len(maybe_nii) > 1:
            logging.warning(
                f"Unexpected number of  *T1w.nii.gz found in {tmpd_in}: {maybe_nii}. Taking first."
            )
        nii = maybe_nii[0]
        nii_out = tmpd_out / nii.stem

        logging.info(f"Staging and unzipping {nii}")
        with gzip.open(nii, "rb") as f_in:
            with open(nii_out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if is_oblique(nii_out):
            logging.warning(f"Detected oblique {nii_out=}")
            returncode = await self.deoblique(nii_out)
            if not returncode == 0:
                logging.error(f"Failed to deoblique {nii_out=}")
                return

        return nii_out

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:

        staged_in = await self.prep(tmpd_in, tmpd_out)
        if staged_in:
            async with utils.subprocess_manager(
                log=tmpd_out / f"brainager_rank-{self.RANK}.log",
                args=get_brainager_args(nii=staged_in),
            ) as proc:
                await proc.wait()
