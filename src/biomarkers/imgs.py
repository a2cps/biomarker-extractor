import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
import pydantic
from nibabel import processing
from nilearn import image, masking, signal
from scipy import ndimage
from skimage import morphology
from sklearn import decomposition

from biomarkers import utils
from biomarkers.models import bids

COMPCOR_LABEL: typing.TypeAlias = typing.Literal["WM", "CSF", "WM+CSF"]


class CompCor(pydantic.BaseModel):
    img: pydantic.FilePath
    probseg: bids.ProbSeg
    label: COMPCOR_LABEL
    boldref: pydantic.FilePath
    high_pass: float | None = None
    low_pass: float | None = None
    n_non_steady_state_tr: int = 0
    detrend: bool = False

    @property
    def boldref_nii(self) -> nb.nifti1.Nifti1Image:
        return nb.nifti1.Nifti1Image.load(self.boldref)

    def fit_transform(self) -> pl.DataFrame:
        """
        general strategy:
        - detrend timeseries
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

        return self.components

    @property
    def components(self) -> pl.DataFrame:
        X: np.ndarray = masking.apply_mask(imgs=self.img, mask_img=self.acompcor_mask)
        if self.detrend:
            X = _detrend(X)

        tr = utils.get_tr(nb.nifti1.Nifti1Image.load(self.img))
        sample_mask = utils.exclude_to_index(
            n_non_steady_state_tr=self.n_non_steady_state_tr, n_tr=X.shape[0]
        )
        X_cleaned: np.ndarray = signal.clean(
            X,
            detrend=False,
            standardize="zscore_sample",
            standardize_confounds="zscore_sample",  # type:ignore
            high_pass=self.high_pass,
            low_pass=self.low_pass,
            t_r=tr,
            sample_mask=[utils.int_sample_mask_to_bool(sample_mask)],
            extrapolate=False,
        )
        del X

        # compcor works on PCA of MM^T
        return self.comp_cor(X=X_cleaned.T, t_index=sample_mask)

    def comp_cor(
        self, X: np.ndarray, t_index: np.typing.NDArray[np.uint32] | None = None
    ) -> pl.DataFrame:
        if not X.ndim == 2:
            msg = "y must be a 2D array"
            raise AssertionError(msg)
        if not X.shape[0] >= X.shape[1]:
            msg = (
                "looks like fewer samples (voxels) than features (volumes). transposed?"
            )
            raise AssertionError(msg)

        # need all components for explained_variance_ratio_ to be accurate
        pca = decomposition.PCA()
        pca.fit(X)
        d = pl.DataFrame(pca.components_.T)
        if t_index is None:
            d = d.with_row_index("t")
        else:
            d = d.with_columns(t=t_index)  # type: ignore

        # keep all components
        return d.unpivot(index="t", variable_name="component").with_columns(
            pl.col("component").str.extract(r"(\d+)").cast(pl.UInt16)
        )

    @property
    def acompcor_mask(self) -> nb.nifti1.Nifti1Image:
        match self.label:
            case "CSF":
                mask = self.probseg.csf_nii
            case "WM":
                mask = self.probseg.wm_nii
            case "WM+CSF":
                mask = self.probseg.gm_nii

        mask_data = np.asarray(mask.dataobj, dtype=np.bool_)

        if "CSF" not in self.label:
            # Dilate the GM mask
            gm_dilated = ndimage.binary_dilation(
                self.probseg.gm_nii.get_fdata() > 0.05, structure=morphology.ball(3)
            )
            # subtract dilated gm from mask to make sure voxel does not contain GM
            mask_data[gm_dilated] = 0

        # Resample probseg maps to BOLD resolution
        # assume already in matching space
        weights_nii = processing.resample_from_to(
            from_img=nb.nifti1.Nifti1Image(
                mask_data, self.probseg.gm_nii.affine, self.probseg.gm_nii.header
            ),
            to_vox_map=self.boldref_nii,
            order=1,
        )
        return nb.nifti1.Nifti1Image(
            weights_nii.get_fdata() > 0.99, weights_nii.affine, weights_nii.header
        )


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
        confounds = pl.read_parquet(confounds_file).to_numpy()
    else:
        confounds = None
    nii = nb.nifti1.Nifti1Image.load(img).slicer[:, :, :, n_non_steady_state_tr:]

    assert len(nii.shape) == 4
    t_r = utils.get_tr(nii)

    if do_detrend:
        nii = detrend(nii, mask=mask)
        if confounds is not None:
            # this does not center confounds, but that is handled
            # during nilearn.image.clean_img
            confounds = _detrend(confounds)
    if do_winsorize:
        nii = winsorize(nii, mask=mask)
    if to_percentchange:
        nii = to_local_percent_change(nii)

    nii: nb.nifti1.Nifti1Image = image.clean_img(
        imgs=nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=t_r,
        confounds=confounds,
        standardize=standardize,
        detrend=False,
        mask_img=mask,
        clean__extrapolate=False,
        clean__standardize_confounds="zscore_sample",
    )  # type: ignore

    if fwhm:
        nii: nb.nifti1.Nifti1Image = image.smooth_img(nii, fwhm=fwhm)  # type: ignore
        return nii
    else:
        return nii


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


def winsorize(
    img: nb.nifti1.Nifti1Image, mask: Path, std: float = 3
) -> nb.nifti1.Nifti1Image:
    dataobj = masking.apply_mask(img, mask_img=mask)

    ms = dataobj.mean(axis=0, keepdims=True)
    stds = dataobj.std(axis=0, ddof=1, keepdims=True)

    # working to get Z to boolean before creating replacements
    where = np.asarray(stds > 0, dtype=np.bool)
    Z = np.zeros_like(dataobj)
    np.subtract(dataobj, ms, out=Z, where=where)
    np.divide(Z, stds, out=Z, where=where)
    np.absolute(Z, out=Z, where=where)
    where_to_replace = np.asarray(Z > std, dtype=np.bool)
    del Z
    replacements = ms + std * stds * np.sign(dataobj - ms)

    dataobj[where_to_replace] = replacements[where_to_replace]
    del replacements

    return masking.unmask(dataobj, mask_img=mask)  # type: ignore


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


def _detrend(Y: np.ndarray, degree: int = 3) -> np.ndarray:
    X = get_poly_design(Y.shape[0], degree=degree)
    beta = np.linalg.pinv(X).dot(Y)
    return Y - np.dot(X[:, 1:], beta[1:, :])


def image_difference(img1: Path, img2: Path) -> nb.nifti1.Nifti1Image:
    # not using nilearn.image.math_img due to memory inefficiency
    active = nb.nifti1.Nifti1Image.load(img1)
    baseline = nb.nifti1.Nifti1Image.load(img2)
    return nb.nifti1.Nifti1Image(
        active.get_fdata() - baseline.get_fdata(),
        header=active.header,
        affine=active.affine,
    )
