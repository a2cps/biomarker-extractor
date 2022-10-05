from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import re

import numpy as np

import nibabel as nb
import pandas as pd

from sklearn import covariance

from nilearn.maskers import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure

import prefect
from prefect.tasks import task_input_hash

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
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),
        kind="correlation",
    )
    correlation_matrix = connectivity_measure.fit_transform([time_series]).squeeze()
    df = _mat_to_df(correlation_matrix, rois.keys()).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )
    df["connectivity"] = np.arctanh(df["connectivity"])
    return df


@prefect.task
def update_confounds(
    acompcor: pd.DataFrame,
    confounds: Path,
    usecols: list[str] = [
        "trans_x",
        "trans_x_derivative1",
        "trans_x_power2",
        "trans_x_derivative1_power2",
        "trans_y",
        "trans_y_derivative1",
        "trans_y_power2",
        "trans_y_derivative1_power2",
        "trans_z",
        "trans_z_derivative1",
        "trans_z_power2",
        "trans_z_derivative1_power2",
        "rot_x",
        "rot_x_derivative1",
        "rot_x_power2",
        "rot_x_derivative1_power2",
        "rot_y",
        "rot_y_derivative1",
        "rot_y_power2",
        "rot_y_derivative1_power2",
        "rot_z",
        "rot_z_derivative1",
        "rot_z_power2",
        "rot_z_derivative1_power2",
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


@dataclass(frozen=True)
class ConnectivityFiles:
    bold: Path
    boldref: Path
    probseg: list[Path]
    confounds: Path


@prefect.task
def get_files(sub: Path, space: str) -> list[ConnectivityFiles]:
    out = []
    s = re.search(r"(?<=sub-)\d{5}", str(sub)).group(0)
    for ses in sub.glob("ses*"):
        e = re.search(r"(?<=ses-)\w{2}", str(ses)).group(0)
        func = ses / "func"
        for run in ["1", "2"]:
            bold = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
            )
            boldref = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_space-{space}_boldref.nii.gz"
            )
            probseg = [x for x in ses.glob(f"anat/*{space}*probseg*")]
            confounds = (
                func
                / f"sub-{s}_ses-{e}_task-rest_run-{run}_desc-confounds_timeseries.tsv"
            )
            if (
                bold.exists()
                and boldref.exists()
                and confounds.exists()
                and all([x.exists() for x in probseg])
            ):
                out.append(
                    ConnectivityFiles(
                        bold=bold, boldref=boldref, probseg=probseg, confounds=confounds
                    )
                )

    return out


@prefect.flow(
    task_runner=DaskTaskRunner(
        cluster_kwargs={"n_workers": 10, "threads_per_worker": 5}
    ),
    validate_parameters=False,
)
def connectivity_flow(
    fmriprep_dir: Path,
    out: Path,
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_seconds: float = 15,
    detrend=False,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    for s in fmriprep_dir.glob("sub-*"):
        to_process = get_files.submit(sub=s, space=space)
        for files in to_process.result():
            acompcor = compcor.do_compcor.submit(
                img=files.bold,
                boldref=files.boldref,
                probseg=files.probseg,
                high_pass=high_pass,
                low_pass=low_pass,
                n_non_steady_state_seconds=n_non_steady_state_seconds,
                detrend=detrend,
            )

            final_confounds = update_confounds.submit(
                acompcor=acompcor,
                confounds=files.confounds,
            )

            task_utils.write_tsv.submit(
                dataframe=acompcor,
                filename=(out / f"{utils.img_stem(files.bold)}_acompcor").with_suffix(
                    ".tsv"
                ),
            )

            connectivity = spheres_connectivity.submit(
                img=files.bold,
                confounds=final_confounds,
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
            )
            task_utils.write_tsv.submit(
                dataframe=connectivity,
                filename=(
                    out / f"{utils.img_stem(files.bold)}_connectivity"
                ).with_suffix(".tsv"),
            )
            task_utils.write_tsv.submit(
                dataframe=final_confounds,
                filename=(out / f"{utils.img_stem(files.bold)}_confounds").with_suffix(
                    ".tsv"
                ),
            )
