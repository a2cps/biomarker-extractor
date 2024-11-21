import typing
from importlib import resources
from pathlib import Path

import polars as pl

NPSWeights: typing.TypeAlias = typing.Literal[
    "FDR05_negative_smoothed_larger_than_10vox",
    "FDR05_positive_smoothed_larger_than_10vox",
    "grouppred_cvpcr_FDR05_smoothed_fwhm05",
    "grouppred_cvpcr_FDR05",
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
]

SIIPS1Weights: typing.TypeAlias = typing.Literal[
    "137subjmap_weighted_mean",
    "137subjmap_weighted_pvalue",
    "subcluster_maps_fdr05_pattern_wttest",
    "subcluster_maps_fdr05_unique_wttest",
]

FAN_RESOLUTION: typing.TypeAlias = typing.Literal["2mm", "3mm"]


def get_nps(weights: NPSWeights) -> Path:
    fname = f"weights_NSF_{weights}.nii.gz"

    with resources.as_file(
        resources.files("biomarkers.data.2013_Wager_NEJM_NPS").joinpath(fname)
    ) as f:
        path = f
    return path


def get_siips1(weights: SIIPS1Weights) -> Path:
    fname = f"nonnoc_v11_4_{weights}.nii.gz"

    with resources.as_file(
        resources.files("biomarkers.data.2017_Woo_SIIPS1").joinpath(fname)
    ) as f:
        path = f
    return path


def get_mpfc_mask() -> Path:
    """Return mPFC mask produced from Smallwood et al. replication.

    Returns:
        Path: Path to mPFC mask.
    """
    with resources.as_file(
        resources.files("biomarkers.data").joinpath("smallwood_mpfc_MNI152_1p5.nii.gz")
    ) as f:
        mpfc = f
    return mpfc


def get_fan_atlas_file(resolution: FAN_RESOLUTION = "2mm") -> Path:
    """Return file from ToPS model (https://doi.org/10.1038/s41591-020-1142-7)

    Returns:
        Path: Path to atlas
    """
    with resources.as_file(
        resources.files("biomarkers.data").joinpath(
            f"Fan_et_al_atlas_r279_MNI_{resolution}.nii.gz"
        )
    ) as f:
        atlas = f
    return atlas


def get_power2011_coordinates_file() -> Path:
    """Return file for volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        Path: Path to atlas
    """
    with resources.as_file(
        resources.files("biomarkers.data").joinpath("power2011_atlas.tsv")
    ) as f:
        atlas = f
    return atlas


def get_mni6gray_mask() -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath("MNI152_T1_6mm_gray.nii.gz")
    ) as f:
        out = f
    return out


def get_power2011_coordinates() -> pl.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    return pl.read_csv(get_power2011_coordinates_file(), separator="\t")


def get_cat_batch() -> Path:
    with resources.path("biomarkers.data", "batch.m") as f:
        mpfc = f
    return mpfc
