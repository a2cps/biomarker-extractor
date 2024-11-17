import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
from nibabel import processing
from nilearn import image, masking

from biomarkers import utils


@utils.cache_nii
def correct_bias(
    img: Path,
    mask: Path,
    s: int | None = None,
    b: float | str | None = None,
    c: int | str | None = None,
) -> nb.nifti1.Nifti1Image:
    import os

    image = nb.nifti1.Nifti1Image.load(img)
    avg = nb.nifti1.Nifti1Image(
        dataobj=image.get_fdata().mean(-1), affine=image.affine, header=image.header
    )
    args = ""
    if s:
        args += f" -s {s}"
    if b:
        args += f" -b [ {b} ]"
    if c:
        args += f" -c [ {c} ]"

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        avg.to_filename(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as o:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz") as biasfile:
                os.system(
                    f"N4BiasFieldCorrection -i {f.name} -o [ {o.name},{biasfile.name} ] -x {mask} {args}"
                )
                bias_field: np.ndarray = nb.nifti1.load(biasfile.name).get_fdata()

    debiased = image.get_fdata()
    for tr in range(debiased.shape[-1]):
        debiased[:, :, :, tr] /= bias_field

    return nb.nifti1.Nifti1Image(
        dataobj=debiased, affine=image.affine, header=image.header
    )


@utils.cache_nii
def clean_img(
    img: Path,
    mask: Path,
    confounds_file: Path | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    do_detrend: bool = False,
    standardize: bool = False,
    fwhm: float
    | np.ndarray
    | tuple[float]
    | list[float]
    | typing.Literal["fast"]
    | None = None,
    do_winsorize: bool = False,
    to_percentchange: bool = False,
    n_non_steady_state_tr: int = 0,
) -> nb.nifti1.Nifti1Image:
    if confounds_file:
        confounds = pl.read_parquet(confounds_file).to_pandas()
    else:
        confounds = None
    nii = nb.nifti1.Nifti1Image.load(img).slicer[:, :, :, n_non_steady_state_tr:]

    assert len(nii.shape) == 4

    if do_detrend:
        nii = detrend(nii, mask=mask)
    if do_winsorize:
        nii = winsorize(nii)
    if to_percentchange:
        nii = to_local_percent_change(nii)

    # note that this relies on default behavior for standardizing confounds when passed to image.clean
    nii_clean: nb.nifti1.Nifti1Image = image.clean_img(
        imgs=nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        confounds=confounds,
        standardize=standardize,
        detrend=False,
        mask_img=mask,
    )  # type: ignore

    if fwhm:
        nii_smoothed: nb.nifti1.Nifti1Image = image.smooth_img(nii_clean, fwhm=fwhm)  # type: ignore
        return nii_smoothed
    else:
        return nii_clean


def to_local_percent_change(
    img: nb.nifti1.Nifti1Image, fwhm: float = 16
) -> nb.nifti1.Nifti1Image:
    avg = nb.nifti1.Nifti1Image(
        dataobj=img.get_fdata().mean(-1), affine=img.affine, header=img.header
    )
    smoothed = processing.smooth_image(avg, fwhm=fwhm)
    pc = np.asarray(img.get_fdata().copy())
    for tr in range(img.shape[-1]):
        pc[:, :, :, tr] -= avg.get_fdata()
        pc[:, :, :, tr] /= smoothed.get_fdata()
    pc *= 100
    pc += 100

    return nb.nifti1.Nifti1Image(dataobj=pc, affine=img.affine, header=img.header)


def winsorize(img: nb.nifti1.Nifti1Image, std: float = 3) -> nb.nifti1.Nifti1Image:
    # from scipy.stats import mstats
    # from scipy import stats

    ms = img.get_fdata().mean(axis=-1, keepdims=True)
    stds = img.get_fdata().std(axis=-1, ddof=1, keepdims=True)

    Z = np.abs((img.get_fdata() - ms) / stds)
    # Z = np.abs(stats.zscore(img.get_fdata(), axis=-1, ddof=1))
    if (Z > std).mean() > 0.01:
        raise ValueError("We're removing more than 1% of values!")

    replacements = ms + std * stds * np.sign(img.get_fdata() - ms)
    winsorized = img.get_fdata().copy()
    winsorized[Z > std] = replacements[Z > std]

    # winsorized = mstats.winsorize(img.get_fdata(), limits=[lower, upper], axis=-1)

    return nb.nifti1.Nifti1Image(
        dataobj=winsorized, affine=img.affine, header=img.header
    )


def get_poly_design(N: int, degree: int) -> np.ndarray:
    x = np.arange(N)
    x = x - np.mean(x, axis=0)
    X = np.vander(x, degree, increasing=True)
    q, r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    Z = raw / np.sqrt(norm2)
    return Z


def detrend(img: nb.nifti1.Nifti1Image, mask: Path) -> nb.nifti1.Nifti1Image:
    Y = masking.apply_mask(img, mask_img=mask)

    resid = _detrend(Y=Y)
    # Put results back into Niimg-like object
    return masking.unmask(resid, mask)  # type: ignore


def _detrend(Y: np.ndarray) -> np.ndarray:
    X = get_poly_design(Y.shape[0], degree=3)
    beta = np.linalg.pinv(X).dot(Y)
    return Y - np.dot(X[:, 1:], beta[1:, :])
