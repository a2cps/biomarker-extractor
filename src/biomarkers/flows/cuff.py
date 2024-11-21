import re
from pathlib import Path
from typing import Literal

import nibabel as nb
import numpy as np
import pandas as pd
import prefect
import pydantic
from nibabel import processing
from nilearn import masking
from pydantic.dataclasses import dataclass
from scipy import ndimage, spatial, stats

# from prefect.tasks import task_input_hash
from .. import utils
from ..task import imgs
from ..task import utils as task_utils

# from sklearn.decomposition import PCA

# from sklearn.covariance import EmpiricalCovariance
# from scipy import stats


@dataclass(frozen=True)
class ImgFiles:
    bold: pydantic.FilePath
    probseg: frozenset[pydantic.FilePath]
    boldref: pydantic.FilePath
    confounds: pydantic.FilePath
    mask: pydantic.FilePath
    stem: str


def _cor(x, y) -> np.float64:
    return stats.pearsonr(x, y).statistic  # type: ignore


def _dot(x, y) -> np.float64:
    return np.inner(x, y).astype(np.float64)


def _cos(x, y) -> np.float64:
    return 1 - spatial.distance.cosine(x, y)


def _add_parts_inplace(d: pd.DataFrame) -> None:
    d["part"] = "beginning"
    d.loc[d["tr"].between(150 - 12, 299 - 12), "part"] = "middle"
    d.loc[d["tr"].gt(300 - 12), "part"] = "end"


@prefect.task
@utils.cache_dataframe
def apply_regionweights(
    img: Path,
    weights: Path,
    method: list[Literal["dot", "cosine", "correlation"]],
) -> pd.DataFrame:
    nii = nb.load(img)

    # manually do resampling to control interpolation scheme
    weights_raw = nb.load(weights)
    target = nii.slicer[:, :, :, 0]
    weights_nii: nb.Nifti1Image = processing.resample_from_to(
        from_img=weights_raw, to_vox_map=target, order=1
    )
    del weights_raw

    # structure matches fsl cluster 26 connectivity
    labeled: np.ndarray = ndimage.label(  # type: ignore
        weights_nii.get_fdata(),
        structure=ndimage.generate_binary_structure(3, 3),
    )[0]  # type: ignore

    out = []
    # first roi is background
    for roi in np.unique(labeled).tolist()[1:]:
        weights_mask = nb.Nifti1Image(
            dataobj=(labeled == roi).astype("int8"),
            affine=weights_nii.affine,
        )
        img_labeled: np.ndarray = masking.apply_mask(imgs=img, mask_img=weights_mask)
        weights_labeled: np.ndarray = masking.apply_mask(
            imgs=weights_nii, mask_img=weights_mask
        )
        n_tr = img_labeled.shape[0]
        n_voxels = img_labeled.shape[1]
        d = pd.DataFrame.from_dict(
            {
                "tr": np.repeat(range(n_tr), n_voxels),
                "voxel": np.tile(range(n_voxels), n_tr),
                "roi": roi,
                "signal": img_labeled.flatten(
                    order="C"
                ),  # be careful about unraveling order!
            }
        )
        _add_parts_inplace(d)

        # aggregate across parts
        bypart = _apply_weights(
            d,
            groups=["part", "roi"],
            method=method,
            weights_masked=weights_labeled,
        )
        bytr = _apply_weights(
            d,
            groups=["tr", "part", "roi"],
            method=method,
            weights_masked=weights_labeled,
        )

        grandavg = _apply_weights(
            d,
            groups=["roi"],
            method=method,
            weights_masked=weights_labeled,
        ).rename(columns={"signal": "signal_grandavg"})
        out.append(
            bytr.merge(
                bypart,
                how="left",
                on=["method", "part", "roi"],
                suffixes=("_tr", "_part"),
            ).merge(grandavg, how="left", on=["method", "roi"])
        )

    return pd.concat(out).reset_index()


def _apply_weights(
    d: pd.DataFrame,
    method: list[Literal["dot", "cosine", "correlation"]],
    weights_masked: np.ndarray,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    out = []

    # dot product is performing very strangely with pandas 1.15.3
    # unlike all other aggregations, it is throwing a ValueError,
    # Length mismatch: Expected axis has {old_len}, new values
    # have {new_len} elements. So, it doesn't recognize that the
    # return value should be a scalar.
    # but it only happens with the ungrouped dataframe
    if groups:
        grouped = (
            d.groupby(groups + ["voxel"])
            .agg(signal=("signal", np.mean))
            .groupby(groups)
        )
        if "dot" in method:
            out.append(
                grouped.agg(
                    signal=("signal", lambda x: _dot(x, weights_masked))
                ).assign(method="dot")
            )
        if "correlation" in method:
            out.append(
                grouped.agg(
                    signal=("signal", lambda x: _cor(x, weights_masked))
                ).assign(method="correlation")
            )
    else:
        grouped = d.groupby("voxel").agg(signal=("signal", np.mean))
        if "dot" in method:
            out.append(
                pd.DataFrame(
                    {"signal": [_dot(grouped["signal"], weights_masked)]}
                ).assign(method="dot")
            )
        if "correlation" in method:
            out.append(
                pd.DataFrame(
                    {"signal": [_cor(grouped["signal"], weights_masked)]}
                ).assign(method="correlation")
            )
    if "cosine" in method:
        out.append(
            grouped.agg(signal=("signal", lambda x: _cos(x, weights_masked))).assign(
                method="cosine"
            )
        )
    return pd.concat(out).reset_index()


@prefect.task()
@utils.cache_dataframe
def apply_weights(
    img: Path,
    weights: Path,
    method: list[Literal["dot", "cosine", "correlation"]],
    mask: Path | None = None,
) -> pd.DataFrame:
    nii = nb.load(img)

    weights_raw = nb.load(weights)
    target = nii.slicer[:, :, :, 0]
    if len(np.unique(weights_raw.get_fdata())) == 2:
        order = 0
    else:
        order = 1
    weights_nii: nb.Nifti1Image = processing.resample_from_to(
        from_img=weights_raw, to_vox_map=target, order=order
    )

    del weights_raw

    if mask:
        weights_mask: nb.Nifti1Image = nb.load(mask)
    else:
        weights_mask = nb.Nifti1Image(
            dataobj=np.logical_not(np.isclose(weights_nii.get_fdata(), 0)).astype(
                "uint8"
            ),
            affine=weights_nii.affine,
        )

    # n_tr x n_voxels (2d)
    masked: np.ndarray = masking.apply_mask(imgs=img, mask_img=weights_mask)

    # n_voxels (1d)
    weights_masked: np.ndarray = masking.apply_mask(
        imgs=weights_nii, mask_img=weights_mask
    )

    n_tr = masked.shape[0]
    n_voxels = masked.shape[1]
    d = pd.DataFrame.from_dict(
        {
            "tr": np.repeat(range(n_tr), n_voxels),
            "voxel": np.tile(range(n_voxels), n_tr),
            "signal": masked.flatten(order="C"),  # be careful about unraveling order!
        }
    )
    _add_parts_inplace(d)

    # aggregate across parts
    bypart = _apply_weights(
        d,
        groups=["part"],
        method=method,
        weights_masked=weights_masked,
    )
    bytr = _apply_weights(
        d,
        groups=["tr", "part"],
        method=method,
        weights_masked=weights_masked,
    )

    grandavg = (
        _apply_weights(
            d,
            method=method,
            weights_masked=weights_masked,
        )
        .rename(columns={"signal": "signal_grandavg"})
        .drop(columns=["index"], axis=1)
    )
    return (
        bytr.merge(
            bypart,
            how="left",
            on=["method", "part"],
            suffixes=("_tr", "_part"),
        )
        .merge(grandavg, how="left", on=["method"])
        .assign(n_voxel=n_voxels, n_tr=n_tr)
    )

    # return out.join(grandavg.set_index("part"), on=["part"], how="left", lsuffix="_tr")
    # return out


# @prefect.task(cache_key_fn=task_input_hash)
@prefect.task
def get_imgs(sub: Path, space: str) -> frozenset[ImgFiles]:
    out = set()
    subgroup = re.search(r"(?<=sub-)\d{5}", str(sub))
    if subgroup is None:
        raise ValueError(f"{sub=} doesn't look like a bids sub directory")
    else:
        s = subgroup.group(0)
    for ses in sub.glob("ses*"):
        sesgroup = re.search(r"(?<=ses-)\w{2}", str(ses))
        if sesgroup is None:
            raise ValueError(f"{ses=} doesn't look like a bids ses directory")
        else:
            e = sesgroup.group(0)
        func = ses / "func"
        for task in ["cuff", "rest"]:
            for run in ["1", "2"]:
                bold = (
                    func
                    / f"sub-{s}_ses-{e}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
                )
                probseg = frozenset(ses.glob(f"anat/*{space}*probseg*"))
                confounds = (
                    func
                    / f"sub-{s}_ses-{e}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
                )
                boldref = (
                    func
                    / f"sub-{s}_ses-{e}_task-{task}_run-{run}_space-{space}_boldref.nii.gz"
                )
                mask = (
                    func
                    / f"sub-{s}_ses-{e}_task-{task}_run-{run}_space-{space}_desc-brain_mask.nii.gz"
                )
                if (
                    bold.exists()
                    and boldref.exists()
                    and confounds.exists()
                    and mask.exists()
                    and all([x.exists() for x in probseg])
                ):
                    out.add(
                        ImgFiles(
                            bold=bold,
                            boldref=boldref,
                            probseg=probseg,
                            confounds=confounds,
                            mask=mask,
                            stem=utils.img_stem(bold),
                        )
                    )

    return frozenset(out)


# def get_mahalanobis(X: np.ndarray) -> np.ndarray:
#     # n_tr x n_tr
#     # covariance = np.cov(X.T, bias=1, dtype=np.float32)
#     pca = PCA(whiten=True)
#     X_transformed = pca.fit_transform(X)

#     # emp_cov = EmpiricalCovariance().fit(X)
#     # Calculate Mahalanobis distances for samples
#     # sqrt because mahalanobis returns squared distances
#     # emp_mahal = np.sqrt(emp_cov.mahalanobis(X))


# @prefect.task
# @utils.cache_dataframe
# def get_outliers(
#     img: Path,
#     mask: Path,
#     mahalanobis_sd: float | None = None,
#     n_non_steady_state_tr: int = 0,
# ) -> pd.DataFrame:

#     # n_tr x n_voxels
#     X: np.ndarray = masking.apply_mask(imgs=img, mask_img=mask)[
#         n_non_steady_state_tr:, :
#     ]

#     spikes = np.zeros(X.shape[0])

#     # https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html
#     if mahalanobis_sd:
#         dist = get_mahalanobis(X)
#         spikes[np.abs(stats.zscore(dist)) > mahalanobis_sd] = 1

#     return pd.DataFrame.from_dict({"spikes": spikes})


@prefect.flow
def cuff_flow(
    subdirs: frozenset[Path],
    out: Path,
    high_pass: float | None = None,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    detrend: bool = False,
    space: str = "MNI152NLin2009cAsym",
    fwhm: float | None = None,
    winsorize: bool = False,
    to_percentchange: bool = True,
    method: list[Literal["dot", "cosine", "correlation"]] = [
        "dot",
        "cosine",
        "correlation",
    ],
    # mahalanobis_sd: float | None = 3,
) -> None:
    nps = utils.get_nps_mask("group")
    nps_binary = utils.get_nps_mask("binary")
    # pos = utils.get_nps_mask("positive")
    # neg = utils.get_nps_mask("negative")
    # roi = utils.get_nps_mask("rois")

    for subdir in subdirs:
        # not submitting to enable acess of individual parts
        for file in get_imgs.submit(sub=subdir, space=space).result():
            # outliers = get_outliers.submit(
            #     out / "outliers" / f"img={file.stem}/part-0.parquet",
            #     img=file.bold,
            #     mask=file.mask,
            #     mahalanobis_sd=mahalanobis_sd,
            #     n_non_steady_state_tr=12,
            # )
            for percentchange in [True, False]:
                final_confounds = task_utils.update_confounds.submit(
                    out / "confounds" / f"img={file.stem}/part-0.parquet",
                    # acompcor_file=acompcor,  # type: ignore
                    confounds=file.confounds,
                    # label=None,
                    n_non_steady_state_tr=n_non_steady_state_tr,
                    # extra=outliers,  # type: ignore
                )

                nii_clean = imgs.clean_img.submit(
                    out
                    / "cleaned"
                    / f"percentchange={percentchange}"
                    / f"{file.stem}.nii.gz",
                    img=file.bold,  # type: ignore
                    confounds_file=final_confounds,  # type: ignore
                    high_pass=high_pass,
                    low_pass=low_pass,
                    detrend=detrend,
                    mask=file.mask,
                    fwhm=fwhm,
                    winsorize=winsorize,
                    to_percentchange=percentchange,
                )

                apply_weights.submit(
                    out
                    / "nps"
                    / "filter=low"
                    / "debias=false"
                    / f"percentchange={percentchange}"
                    / f"img={file.stem}/part-0.parquet",
                    img=nii_clean,  # type: ignore
                    weights=nps,
                    method=method,
                )

                apply_weights.submit(
                    out
                    / "nps-binary"
                    / "filter=low"
                    / "debias=false"
                    / f"percentchange={percentchange}"
                    / f"img={file.stem}/part-0.parquet",
                    img=nii_clean,  # type: ignore
                    weights=nps_binary,
                    method=["dot"],
                )

                # now, repeat process for each region!
                # apply_regionweights.submit(
                #     out
                #     / "nps-roi"
                #     / "filter=low"
                #     / "debias=false"
                #     / f"percentchange={percentchange}"
                #     / f"img={file.stem}/part-0.parquet",
                #     img=nii_clean,  # type: ignore
                #     weights=roi,
                #     method=method,
                # )

                # for (f)ALFF
                nofilter = imgs.clean_img.submit(
                    out
                    / "cleaned-nofilter"
                    / f"percentchange={percentchange}"
                    / f"{file.stem}.nii.gz",
                    img=file.bold,  # type: ignore
                    confounds_file=final_confounds,  # type: ignore
                    high_pass=None,
                    low_pass=None,
                    detrend=detrend,
                    mask=file.mask,
                    fwhm=fwhm,
                    winsorize=winsorize,
                    to_percentchange=percentchange,
                )
                apply_weights.submit(
                    out
                    / "nps"
                    / "filter=none"
                    / "debias=false"
                    / f"percentchange={percentchange}"
                    / f"img={file.stem}/part-0.parquet",
                    img=nofilter,  # type: ignore
                    weights=nps,
                    method=method,
                )
                # apply_regionweights.submit(
                #     out
                #     / "nps-roi"
                #     / "filter=none"
                #     / "debias=false"
                #     / f"percentchange={percentchange}"
                #     / f"img={file.stem}/part-0.parquet",
                #     img=nofilter,  # type: ignore
                #     weights=roi,
                #     method=method,
                # )


# # now again, but with debiasing!?
# debiased = imgs.correct_bias.submit(
#     out / "debiased" / f"{file.stem}.nii.gz", img=file.bold, mask=file.mask
# )

# nii_clean_debiased = imgs.clean_img.submit(
#     out / "cleaned-debiased" / f"{file.stem}.nii.gz",
#     img=debiased,  # type: ignore
#     confounds_file=final_confounds,  # type: ignore
#     high_pass=high_pass,
#     low_pass=low_pass,
#     detrend=detrend,
#     mask=file.mask,
#     fwhm=fwhm,
# )
# apply_weights.submit(
#     out
#     / "nps"
#     / "filter=low"
#     / "debias=true"
#     / f"img={file.stem}/part-0.parquet",
#     img=nii_clean_debiased,  # type: ignore
#     weights=nps,
# )
# # now, repeat process for each region!
# apply_regionweights.submit(
#     out
#     / "nps-roi"
#     / "filter=low"
#     / "debias=true"
#     / f"img={file.stem}/part-0.parquet",
#     img=nii_clean_debiased,  # type: ignore
#     weights=roi,
# )

# # for (f)ALFF
# nofilter_debiased = imgs.clean_img.submit(
#     out / "cleaned-nofilter-debiased" / f"{file.stem}.nii.gz",
#     img=debiased,  # type: ignore
#     confounds_file=final_confounds,  # type: ignore
#     high_pass=None,
#     low_pass=None,
#     detrend=detrend,
#     mask=file.mask,
#     fwhm=fwhm,
# )
# apply_weights.submit(
#     out
#     / "nps"
#     / "filter=none"
#     / "debias=true"
#     / f"img={file.stem}/part-0.parquet",
#     img=nofilter_debiased,  # type: ignore
#     weights=nps,
# )
# apply_regionweights.submit(
#     out
#     / "nps-roi"
#     / "filter=none"
#     / "debias=true"
#     / f"img={file.stem}/part-0.parquet",
#     img=nofilter_debiased,  # type: ignore
#     weights=roi,
# )
