import nibabel as nb
import numpy as np
import polars as pl
import pydantic
from nibabel import processing
from nilearn import _utils, masking

from biomarkers import datasets

NPS: tuple[datasets.NPSWeights, ...] = (
    "grouppred_cvpcr_FDR05_smoothed_fwhm05",
    "grouppred_cvpcr",
    "positive_smoothed_larger_than_10vox",
    "positive_smoothed_larger_than_10vox_roi-dACC",
    "positive_smoothed_larger_than_10vox_roi-lIns",
    "positive_smoothed_larger_than_10vox_roi-rdpIns",
    "positive_smoothed_larger_than_10vox_roi-rIns",
    "positive_smoothed_larger_than_10vox_roi-rS2_Op",
    "positive_smoothed_larger_than_10vox_roi-rThal",
    "positive_smoothed_larger_than_10vox_roi-rV1",
    "positive_smoothed_larger_than_10vox_roi-vermis",
    "negative_smoothed_larger_than_10vox",
    "negative_smoothed_larger_than_10vox_roi-lLOC",
    "negative_smoothed_larger_than_10vox_roi-lSTS",
    "negative_smoothed_larger_than_10vox_roi-PCC",
    "negative_smoothed_larger_than_10vox_roi-pgACC",
    "negative_smoothed_larger_than_10vox_roi-rIPL",
    "negative_smoothed_larger_than_10vox_roi-rLOC",
    "negative_smoothed_larger_than_10vox_roi-rpLOC",
)

SIIPS: tuple[datasets.SIIPS1Weights, ...] = ("137subjmap_weighted_mean",)


class Func3d(pydantic.BaseModel):
    label: str
    path: pydantic.FilePath
    dtype: str = "f8"

    def to_polars(self, target: nb.nifti1.Nifti1Image | None = None) -> pl.DataFrame:
        i: nb.nifti1.Nifti1Image = _utils.check_niimg_3d(self.path)  # type: ignore

        if target:
            img = processing.resample_from_to(i, target, order=1)
        else:
            img = i

        out = masking.apply_mask(img, make_mask(img), dtype=self.dtype)
        return pl.DataFrame(
            {"voxel": np.arange(out.shape[0], dtype=np.uint32), self.label: out}
        )


class Func4d(pydantic.BaseModel):
    path: pydantic.FilePath
    dtype: str = "f8"
    label: str = "signal"

    def to_polars(self) -> pl.DataFrame:
        """Convert 4D nifti image into polars dataframe

        Returns:
            pl.DataFrame: data from input img
        """

        i: nb.nifti1.Nifti1Image = _utils.check_niimg(self.path, ensure_ndim=4)  # type: ignore
        out = masking.apply_mask(i, make_mask(i), dtype=self.dtype)
        d = (
            pl.DataFrame(out, schema=[str(x) for x in range(out.shape[1])])
            .with_columns(pl.Series("t", np.arange(out.shape[0], dtype=np.uint16)))
            .unpivot(index=["t"], value_name=self.label, variable_name="voxel")
            .with_columns(pl.col("voxel").cast(pl.UInt32()))
        )

        return d


def make_mask(img: nb.nifti1.Nifti1Image) -> nb.nifti1.Nifti1Image:
    """Make mask that can be used as no-op.

    Args:
        img nb.Nifti1Image: Image whose shape and affine will be used to make mask.

    Returns:
        nb.Nifti1Image: Image (uint8) of size input with all values equal to 1
    """
    return nb.nifti1.Nifti1Image(
        dataobj=np.ones(img.shape[:3], dtype=np.uint8), affine=img.affine
    )
