from fileinput import filename
from pathlib import Path
import numpy as np

# from numpy.polynomial import Legendre
import pandas as pd

from nilearn import signal
from nilearn.masking import apply_mask

from sklearn.decomposition import PCA

import nibabel as nb

import ants

from skimage.morphology import ball
from scipy import ndimage

import prefect

from .. import utils

# TODO:
# - detrend (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.Legendre.fit.html#numpy.polynomial.legendre.Legendre.fit)
# - check affine of mask

"""
general strategy: 
- censor timeseries
- detrend timeseries (grab residuals from .fit) ?
- temporal filter with nilearn.signals.clean / nilearn.signals.butterworth (and standardize, to acheive columnwise variance normalization)
- calculate components from resulting timseries 
  - nilearn.signals.high_variance_confounds (https://github.com/nilearn/nilearn/blob/2173571e7d8896e562575a28baec681c4785cbef/nilearn/signal.py#L385)
  - nipype.algorithms.confounds.compute_noise_components (https://github.com/nipy/nipype/blob/b1cc5b681d6980d725c39dd6274808bb95d58bc5/nipype/algorithms/confounds.py#L1326)

Voxel time series from the noise ROI (either anatomical or tSTD) were placed in a matrix M of size 
N x m, with time along the row dimension and voxels along the column dimension. 
The constant and linear trends of the columns in the matrix M were removed prior to column-wise 
variance normalization. The covariance matrix C = MMT was constructed and decomposed into its 
principal components using a singular value decomposition.
"""


# def _expected_largest_piece(piece: int, n_pieces: int) -> float:
# https://blogs.sas.com/content/iml/2017/08/02/retain-principal-components.html
# return np.divide(1, np.arange(piece + 1, n_pieces + 1)).sum() / n_pieces


# def _detrend_voxel(y: np.ndarray, deg: int = 2) -> np.ndarray:
#     # should this have any upsampling?
#     assert y.ndim == 1, "y must be a 1D array"
#     n_tr = y.shape[0]
#     x = np.arange(n_tr)
#     legendre = Legendre.fit(x, y, deg=deg)
#     return y - legendre(x)


def comp_cor(X: np.ndarray) -> pd.DataFrame:
    """_summary_

    Args:
        X (np.ndarray): (n_samples, n_features) should already be cleaned. Samples are voxels and features are volumes.

    Raises:
        AssertionError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    assert X.ndim == 2, "y must be a 2D array"
    assert (
        X.shape[0] >= X.shape[1]
    ), "looks like fewer samples (voxels) than features (volumes). transposed?"
    n_tr = X.shape[1]

    # need all components for explained_variance_ratio_ to be accurate
    pca = PCA()
    pca.fit(X)

    # keep all components
    out = (
        pd.DataFrame(pca.components_.T)
        .assign(tr=range(n_tr))
        .melt(id_vars=["tr"], var_name="component")
        .infer_objects()
        .assign(n_samples=pca.n_samples_)
    )
    # compfor files end up as integers
    # need str for eventual writting to parquet
    out.columns = out.columns.astype(str)
    out["explained_variance_ratio"] = pca.explained_variance_ratio_[out["component"]]
    # if truncate is None:
    # out = pd.DataFrame(pca.components_.T)
    # keep only components that pass broken stick threshold
    # elif truncate == "stick":
    #     kept = []
    #     for c, ratio in enumerate(pca.explained_variance_ratio_):
    #         if ratio > _expected_largest_piece(c, n_tr):
    #             kept.append(c)
    #         else:
    #             break
    #     # components stored as rows, so transpose
    #     out = pd.DataFrame(pca.components_[kept, :].T)
    return out


def get_components(
    img: Path,
    mask: nb.Nifti1Image,
    high_pass: float | None = None,
    low_pass: float | None = None,
    n_non_steady_state_seconds: float = 0,
    detrend: bool = False,
) -> pd.DataFrame:

    X: np.ndarray = apply_mask(imgs=img, mask_img=mask)
    tr = utils.get_tr(nb.load(img))
    X_cleaned = signal.clean(
        X,
        detrend=detrend,
        standardize="zscore",
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=tr,
        sample_mask=utils.sec_to_index(
            seconds=n_non_steady_state_seconds, tr=tr, n_tr=X.shape[0]
        ),
    )
    del X

    # compcor works on PCA of MM^T
    return comp_cor(X=X_cleaned.T)


def get_acompcor_mask(
    target: Path,
    gray_matter: Path,
    mask_matters: list[Path],
) -> nb.Nifti1Image:

    # PV maps from FAST
    gm_nii = nb.load(gray_matter)

    mask_data = np.zeros(gm_nii.shape, dtype=np.bool_)
    for mask in mask_matters:
        mask_data |= np.asanyarray(nb.load(mask).dataobj, dtype=np.bool_)
        if not "CSF" in mask.stem:
            # Dilate the GM mask
            gm_dilated = ndimage.binary_dilation(
                gm_nii.get_fdata() > 0.05, structure=ball(3)
            )
            # subtract dilated gm from mask to make sure voxel does not contain GM
            mask_data[gm_dilated] = 0

    # Resample probseg maps to BOLD resolution
    # assume already in matching space

    # ants has difficulty reading and writing
    target_nii = nb.load(target)
    bold_mask = ants.resample_image_to_target(
        image=ants.from_nibabel(
            nb.Nifti1Image(mask_data, gm_nii.affine, gm_nii.header)
        ),
        target=ants.from_nibabel(target_nii),
        interp_type="lanczosWindowedSinc",
    )

    return ants.to_nibabel(bold_mask > 0.99)
    # threshold/binarize
    # binary_bold_mask = bold_mask.numpy() > 0.99
    # return nb.Nifti1Image(
    #     binary_bold_mask, affine=target_nii.affine, header=target_nii.header
    # )


@prefect.task
@utils.cache_dataframe
def do_compcor(
    img: Path,
    boldref: Path,
    probseg: list[Path],
    high_pass: float | None = None,
    low_pass: float | None = None,
    n_non_steady_state_seconds: float = 0,
    detrend: bool = False,
) -> pd.DataFrame:

    compcors = []
    masks = {
        "GM": [x for x in probseg if "GM" in x.stem][0],
        "CSF": [x for x in probseg if "CSF" in x.stem][0],
        "WM": [x for x in probseg if "WM" in x.stem][0],
    }

    for label in [["WM"], ["CSF"], ["WM", "CSF"]]:
        mask = get_acompcor_mask(
            target=boldref,
            gray_matter=masks["GM"],
            mask_matters=[masks["key"] for key in label],
        )

        compcors.append(
            get_components(
                img=img,
                mask=mask,
                high_pass=high_pass,
                low_pass=low_pass,
                n_non_steady_state_seconds=n_non_steady_state_seconds,
                detrend=detrend,
            ).assign(label="+".join(label), src=img.name)
        )

    return pd.concat(compcors)
