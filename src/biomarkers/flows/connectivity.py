from typing import Literal
from pathlib import Path

import numpy as np

import nibabel as nb
import pandas as pd

from ancpbids import BIDSLayout

from nilearn.maskers import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure

import prefect
from prefect.tasks import task_input_hash

# from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner

from .. import utils
from ..task import utils as task_utils
from ..task import compcor


def _mat_to_df(cormat: np.ndarray, labels: list[str]) -> pd.DataFrame:
    z_dict = {}
    for xi, x in enumerate(labels):
        for yi, y in enumerate(labels):
            if yi <= xi:
                continue
            else:
                z_dict.update({f"{x}-{y}": [cormat[xi, yi]]})

    return (
        pd.DataFrame.from_dict(z_dict)
        .melt(var_name="regions", value_name="connectivity")
        .infer_objects()
    )


@prefect.task(cache_key_fn=task_input_hash)
def spheres_connectivity(
    img: Path,
    confounds: pd.DataFrame,
    rois: dict[str, tuple] = {
        "mPFC": (2, 52, -2),
        "rNAc": (10, 12, -8),
        "rInsula": (40, -6, -2),
        "lSMC": (-32, -34, 66),
    },
    radius: int = 5,  # " ... defined as 10-mm spheres centered ..."
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:

    """
    for confounds,
    - drift has been shown to be approximately quadratic, so NDMG uses a second-degree polynomial regressor.
    - Friston24,
    - top 5 principal components
    """

    nii = nb.load(img)
    masker = NiftiSpheresMasker(
        seeds=rois.values(),
        radius=radius,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        standardize_confounds=False,
        detrend=detrend,
    )
    # confounds are already sliced
    n_tr = confounds.shape[0]
    time_series = masker.fit_transform(
        imgs=nii.slicer[:, :, :, -n_tr:],
        confounds=confounds,
    )
    connectivity_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = connectivity_measure.fit_transform([time_series]).squeeze()
    df = _mat_to_df(correlation_matrix, rois.keys()).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )
    df["connectivity"] = np.arctanh(df["connectivity"])
    return df


# @prefect.task
# def do_compcor(img: Path, fmriprepoutput: fmriprep.FMRIPrepOutput) -> pd.DataFrame:

#     wm_mask = get_acompcor_mask(
#         fixed=img,
#         gray_matter=fmriprepoutput.anat.label_GM_probseg,
#         white_matter=fmriprepoutput.anat.label_WM_probseg,
#         transformlist=[],
#     )
#     wm_compcor = compcor.get_components(img=img, mask=wm_mask).assign(
#         label="WM", src=img.name
#     )
#     csf_mask = get_acompcor_mask(
#         fixed=img,
#         gray_matter=fmriprepoutput.anat.label_GM_probseg,
#         csf=fmriprepoutput.anat.label_CSF_probseg,
#         transformlist=[],
#     )
#     csf_compcor = compcor.get_components(img=img, mask=csf_mask).assign(
#         label="CSF", src=img.name
#     )
#     combined_mask = get_acompcor_mask(
#         fixed=img,
#         gray_matter=fmriprepoutput.anat.label_GM_probseg,
#         white_matter=fmriprepoutput.anat.label_WM_probseg,
#         csf=fmriprepoutput.anat.label_CSF_probseg,
#         transformlist=[],
#     )
#     combined_compcor = compcor.get_components(img=img, mask=combined_mask).assign(
#         label="combined", src=img.name
#     )
#     return pd.concat([wm_compcor, csf_compcor, combined_compcor])


@prefect.task
def update_confounds(
    acompcor: pd.DataFrame,
    confounds: Path,
    usecols: list[str] = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
    ],
    label: Literal["CSF", "WM", "WM+CSF"] = "WM+CSF",
) -> pd.DataFrame:
    components = (
        acompcor[["component", "tr", "value", "label"]]
        .query("label==@label and component < 5")
        .drop("label", axis=1)
        .pivot(index="tr", columns=["component"], values="value")
    )
    n_tr = components.shape[0]
    components_df = (
        pd.read_csv(confounds, delim_whitespace=True, usecols=usecols)
        .iloc[-n_tr:, :]
        .reset_index(drop=True)
    )
    return pd.concat([components_df, components], axis=1)


# @prefect.flow(task_runner=SequentialTaskRunner, validate_parameters=False)
@prefect.flow(task_runner=DaskTaskRunner)
def connectivity_flow(
    fmripreplayout: BIDSLayout,
    out: Path,
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_seconds: float = 15,
    detrend=True,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    for sub in fmripreplayout.get_subjects():
        for ses in fmripreplayout.get_sessions(sub=sub):
            for run in fmripreplayout.get_runs(sub=sub, ses=ses):

                bold = Path(
                    fmripreplayout.get(
                        return_type="file",
                        sub=sub,
                        run=run,
                        task="rest",
                        desc="preproc",
                        space=space,
                        extension=".nii.gz",
                    )[0]
                )

                acompcor: pd.DataFrame = compcor.do_compcor.submit(
                    fmripreplayout=fmripreplayout,
                    sub=sub,
                    run=run,
                    ses=ses,
                    space=space,
                    high_pass=high_pass,
                    low_pass=low_pass,
                    n_non_steady_state_seconds=n_non_steady_state_seconds,
                    detrend=detrend,
                )

                task_utils.write_tsv.submit(
                    dataframe=acompcor,
                    filename=(out / f"{utils.img_stem(bold)}_acompcor").with_suffix(
                        ".tsv"
                    ),
                )
                confounds: pd.DataFrame = update_confounds.submit(
                    acompcor=acompcor,
                    confounds=Path(
                        fmripreplayout.get(
                            return_type="file",
                            suffix="timeseries",
                            run="1",
                            extension=".tsv",
                        )[0]
                    ),
                )

                connectivity: pd.DataFrame = spheres_connectivity.submit(
                    img=bold,
                    confounds=confounds,
                    high_pass=high_pass,
                    low_pass=low_pass,
                    detrend=detrend,
                )
                task_utils.write_tsv.submit(
                    dataframe=connectivity,
                    filename=(out / f"{utils.img_stem(bold)}_connectivity").with_suffix(
                        ".tsv"
                    ),
                )
                task_utils.write_tsv.submit(
                    dataframe=confounds,
                    filename=(out / f"{utils.img_stem(bold)}_confounds").with_suffix(
                        ".tsv"
                    ),
                )
