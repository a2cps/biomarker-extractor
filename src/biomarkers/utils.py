from pathlib import Path
from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd

import nibabel as nb


def img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def get_mpfc_mask() -> Path:
    """Return mPFC mask produced from Smallwood et al. replication.

    Returns:
        Path: Path to mPFC mask.
    """
    with resources.path("biomarkers.data", "smallwood_mpfc_MNI152_1p5.nii.gz") as f:
        mpfc = f
    return mpfc


def get_rs2_labels() -> Path:
    with resources.path("biomarkers.data", "TD_label.nii") as f:
        labels = f
    return labels


def get_power2011_coordinates_file() -> Path:
    """Return file for volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        Path: Path to atlas
    """
    with resources.path("biomarkers.data", "power2011.tsv") as f:
        atlas = f
    return atlas


def get_power2011_coordinates() -> pd.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    return pd.read_csv(
        get_power2011_coordinates_file(),
        delim_whitespace=True,
        index_col="ROI",
        dtype={"x": np.float16, "y": np.int16, "z": np.int16},
    )


def get_mni6gray_mask() -> Path:
    with resources.path("biomarkers.data", "MNI152_T1_6mm_gray.nii.gz") as f:
        out = f
    return out


def sec_to_index(seconds: float, tr: float, n_tr: int) -> np.ndarray:
    return np.array([x for x in range(np.floor(seconds * tr).astype(int), n_tr)])


def get_tr(nii: nb.Nifti1Image) -> float:
    return nii.header.get("pixdim")[4]


def probe_cached_file(filename: Path) -> Path | None:
    if filename.exists():
        return filename
