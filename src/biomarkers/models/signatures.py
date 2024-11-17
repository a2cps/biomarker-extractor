import pydantic
from pydantic import dataclasses

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


@dataclasses.dataclass(frozen=True)
class Func3d:
    label: str
    path: pydantic.FilePath
    dtype: str
