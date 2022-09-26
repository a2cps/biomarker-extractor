from pathlib import Path
from importlib import resources

import numpy as np

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


def sec_to_index(seconds: float, tr: float, n_tr: int) -> np.ndarray:
    return np.array([x for x in range(np.floor(seconds * tr).astype(int), n_tr)])


def get_tr(nii: nb.Nifti1Image) -> float:
    return nii.header.get("pixdim")[4]
