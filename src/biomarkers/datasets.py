import typing
import xml
import xml.etree
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from importlib import resources
from pathlib import Path

import polars as pl
import pydantic

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

YeoNetworks: typing.TypeAlias = typing.Literal[7, 17]
SchaeferNROI: typing.TypeAlias = typing.Literal[400]
SchaeferResolution: typing.TypeAlias = typing.Literal[2]

FanResolution: typing.TypeAlias = typing.Literal[2, 3]

DIFUMODimension: typing.TypeAlias = typing.Literal[64, 128, 256, 512, 1024]
DIFUMOResolution: typing.TypeAlias = typing.Literal[2, 3]

GordonResolution: typing.TypeAlias = typing.Literal[1, 2, 3]

GordonSpace: typing.TypeAlias = typing.Literal["MNI", "711-2b"]


class Labels(pydantic.BaseModel):
    labels_img: pydantic.FilePath
    labels: pl.DataFrame

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


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


def get_power2011_coordinates() -> pl.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    return pl.read_csv(get_power2011_coordinates_file(), separator="\t")


def get_cat_batch() -> Path:
    with resources.as_file(resources.files("biomarkers.data").joinpath("batch.m")) as f:
        mpfc = f
    return mpfc


def get_baliki_lut() -> pl.DataFrame:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath("baliki.csv")
    ) as f:
        coordinates = pl.read_csv(f)
    return coordinates


def get_difumo_lut(dimension: DIFUMODimension) -> pl.DataFrame:
    with resources.as_file(
        resources.files(f"biomarkers.data.difumo_atlases.{dimension}").joinpath(
            f"labels_{dimension}_dictionary.csv"
        )
    ) as f:
        labels = pl.read_csv(f)

    return labels


def get_difumo(dimension: DIFUMODimension, resolution_mm: DIFUMOResolution) -> Labels:
    with resources.as_file(
        resources.files(
            f"biomarkers.data.difumo_atlases.{dimension}.{resolution_mm}mm"
        ).joinpath("maps.nii.gz")
    ) as f:
        maps = f
    labels = get_difumo_lut(dimension=dimension).rename({"Component": "region"})
    return Labels(labels_img=maps, labels=labels)


def get_atlas_schaefer_2018_lut(
    n_rois: SchaeferNROI, yeo_networks: YeoNetworks
) -> pl.DataFrame:
    with resources.as_file(
        resources.files("biomarkers.data.schaefer_2018").joinpath(
            f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.txt"
        )
    ) as f:
        labels = pl.read_csv(
            f,
            separator="\t",
            has_header=False,
            new_columns=["region", "label", "r", "g", "b", "a"],
        )
    return labels


def get_atlas_schaefer_2018(
    n_rois: SchaeferNROI,
    resolution_mm: SchaeferResolution,
    yeo_networks: YeoNetworks,
) -> Labels:
    with resources.as_file(
        resources.files("biomarkers.data.schaefer_2018").joinpath(
            f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order_FSLMNI152_{resolution_mm}mm.nii.gz"
        ),
    ) as f:
        maps = f
    labels = get_atlas_schaefer_2018_lut(n_rois=n_rois, yeo_networks=yeo_networks)

    return Labels(labels_img=maps, labels=labels)


def get_gordon_2016_lut() -> pl.DataFrame:
    with resources.as_file(
        resources.files("biomarkers.data.gordon_2016").joinpath("Parcels.tsv")
    ) as f:
        lut = pl.read_csv(f, separator="\t")

    return lut


def get_atlas_gordon_2016(
    resolution_mm: GordonResolution, space: GordonSpace = "MNI"
) -> Labels:
    with resources.as_file(
        resources.files("biomarkers.data.gordon_2016").joinpath(
            f"Parcels_{space}_{resolution_mm}{resolution_mm}{resolution_mm}.nii.gz"
        )
    ) as f:
        labels_img = f
    labels = get_gordon_2016_lut()

    return Labels(labels_img=labels_img, labels=labels)


def get_fan_atlas_lut_file() -> Path:
    with resources.as_file(resources.files("biomarkers.data").joinpath("fan.csv")) as f:
        labels = f
    return labels


def get_fan_atlas_nii_file(resolution: FanResolution = 2) -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath(
            f"Fan_et_al_atlas_r279_MNI_{resolution}mm.nii.gz"
        )
    ) as f:
        atlas = f
    return atlas


def get_fan_atlas_lut() -> pl.DataFrame:
    f = get_fan_atlas_lut_file()
    labels = pl.read_csv(f)
    return labels


def get_fan_atlas(resolution: FanResolution = 2) -> Labels:
    """Return file from ToPS model (https://doi.org/10.1038/s41591-020-1142-7)"""
    atlas = get_fan_atlas_nii_file(resolution=resolution)
    labels = get_fan_atlas_lut()
    return Labels(labels_img=atlas, labels=labels)


def get_s1m1() -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath(
            "MAPP_S1-SLN_cluster_bin_in_MNI152NLin2009cAsym_brain.nii.gz"
        )
    ) as f:
        s1m1 = f
    return s1m1


def get_s1() -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath(
            "desc-smoothed10bystudy999_S1.nii.gz"
        )
    ) as f:
        s1 = f
    return s1


def get_jhu_dti() -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath(
            "JHU-ICBM-labels-1mm_in_MNI152NLin2009cAsym.nii.gz"
        )
    ) as f:
        s1 = f
    return s1


def get_jhu_lut_file() -> Path:
    with resources.as_file(
        resources.files("biomarkers.data").joinpath("JHU-labels.xml")
    ) as f:
        labels = f
    return labels


def get_jhu_lut() -> pl.DataFrame:
    doc = ET.parse(get_jhu_lut_file()).getroot().find("data")
    if not isinstance(doc, xml.etree.ElementTree.Element):
        msg = "Missing JHU LUT"
        raise AssertionError(msg)
    labels = {"index": [], "roi": []}
    for label in doc[1:]:
        labels["index"].append(int(label.attrib["index"]))
        labels["roi"].append(label.text)

    return pl.DataFrame(labels, schema_overrides={"index": pl.UInt8})


def get_jhu_atlas() -> Labels:
    atlas = get_jhu_dti()
    labels = get_jhu_lut()
    return Labels(labels_img=atlas, labels=labels)
