from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import re

import numpy as np

import nibabel as nb
import pandas as pd

from sklearn import covariance
from sklearn.utils import Bunch

from nilearn import maskers
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets

import networkx as nx
import statsmodels as sm
import patsy

import prefect
from prefect.tasks import task_input_hash

from prefect_dask import DaskTaskRunner


from .. import utils
from ..task import utils as task_utils
from ..task import compcor


# TODO: remove 8 nodes from ower2011 atlas that are in the cerebellum


@dataclass(frozen=True)
class Coordinate:
    label: str
    seed: tuple[int, int, int]


@dataclass(frozen=True)
class ConnectivityFiles:
    bold: Path
    boldref: Path
    probseg: list[Path]
    confounds: Path


def _mat_to_df(cormat: np.ndarray, labels: list[str]) -> pd.DataFrame:
    source = []
    target = []
    connectivity = []
    for xi, x in enumerate(labels):
        for yi, y in enumerate(labels):
            if yi <= xi:
                continue
            else:
                source.append(x)
                target.append(y)
                connectivity.append(cormat[xi, yi])

    return pd.DataFrame.from_dict(
        {"source": source, "target": target, "connectivity": connectivity}
    )


def df_to_coordinates(dataframe: pd.DataFrame) -> list[Coordinate]:
    coordinates = []
    for row in dataframe.itertuples():
        coordinates.append(Coordinate(label=row.label, seed=(row.x, row.y, row.z)))

    return coordinates


@prefect.task
def get_baliki_coordinates() -> list[Coordinate]:
    return [
        Coordinate(label="mPFC", seed=(2, 52, -2)),
        Coordinate(label="rNAc", seed=(10, 12, -8)),
        Coordinate(label="rInsula", seed=(40, -6, -2)),
        Coordinate(label="S1/M1", seed=(-32, -34, 66)),
    ]


@prefect.task
def get_power_coordinates() -> list[Coordinate]:
    rois: pd.DataFrame = datasets.fetch_coords_power_2011(legacy_format=False).rois
    rois.rename(colums={"roi": "label"}, inplace=True)
    return df_to_coordinates(rois)


@prefect.task(cache_key_fn=task_input_hash)
def spheres_connectivity(
    img: Path,
    confounds: pd.DataFrame,
    coordinates: list[Coordinate],
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
    masker = maskers.NiftiSpheresMasker(
        seeds=[x.seed for x in coordinates],
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
    df = _mat_to_df(correlation_matrix, [x.label for x in coordinates]).assign(
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


def get_nedges(n_nodes: int, density: float) -> int:
    return np.floor_divide(density * n_nodes * (n_nodes - 1), 2).astype(int)


def df_to_graph(connectivity: pd.DataFrame, link_density: float) -> nx.Graph:
    g = nx.Graph()
    nodes = set(
        connectivity["source"].unique().tolist()
        + connectivity["target"].unique().tolist()
    )
    g.add_nodes_from(nodes)
    connectivity["connectivity"] = connectivity["connectivity"].abs()
    trimmed = connectivity.nlargest(
        n=get_nedges(n_nodes=len(nodes), density=link_density),
        columns="connectivity",
    )
    g.add_edges_from(trimmed[["source", "target"]].itertuples(index=False))
    return g


@prefect.task
def get_degree(connectivity: pd.DataFrame, src: Path) -> pd.DataFrame:

    degrees = []
    # careful to avoid counting by 0.01
    for density in np.arange(2, 11, 1) / 100:
        g = df_to_graph(connectivity, link_density=density)
        degrees.append(
            pd.DataFrame.from_dict(dict(g.degree), orient="index", columns=["degree"])
            .reset_index()
            .rename(columns={"index": "roi"})
            .assign(density=density)
        )

    return pd.concat(degrees, ignore_index=True).assign(src=src.name)


@prefect.task
def get_schaesrcfer_atlas() -> Bunch:
    return datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=7, resolution_mm=2
    )


@prefect.task
def get_hub_disruption(degrees: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat(degrees, ignore_index=True)
    avgs = d.groupby("roi").agg({"degree": "mean"}).rename(columns={"degree": "avg"})
    d = d.join(avgs, on="roi")
    d["degree_centered"] = d["degree"] - d["avg"]

    # this could be refactored to run in parallel (e.g., convert this task to a flow),
    # but I'm banking on the fitting being quick enough that it won't matter
    hub_disruption = []
    for name, group in d.groupby("src"):
        y, X = patsy.dmatrices(
            "degree_centered ~ avg", data=group, return_type="dataframe"
        )
        model = sm.OLS(y, X)
        fit = model.fit()
        hub_disruption.append(
            pd.DataFrame(
                {
                    "slope": [fit.params.avg],
                    "src": [name],
                }
            )
        )

    return pd.concat(hub_disruption, ignore_index=True)


@prefect.flow(
    task_runner=DaskTaskRunner(
        cluster_kwargs={"n_workers": 10, "threads_per_worker": 5}
    )
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
    acompcor_files = []
    confounds_files = []
    power_files = []
    # degrees_list = []
    baliki_coordinates = get_baliki_coordinates.submit()
    power_coordinates = get_power_coordinates.submit()
    # schaefer_atlas = get_schaefer_atlas.submit()

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

            acompcor_files.append(
                task_utils.write_tsv.with_options(cache_key_fn=None).submit(
                    dataframe=acompcor
                )
            )

            connectivity = spheres_connectivity.submit(
                img=files.bold,
                coordinates=baliki_coordinates,
                confounds=final_confounds,
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
            )

            power_connectivity = spheres_connectivity.submit(
                img=files.bold,
                coordinates=power_coordinates,
                confounds=final_confounds,
                high_pass=high_pass,
                low_pass=low_pass,
                detrend=detrend,
            )

            degrees = get_degree.submit(connectivity=power_connectivity, src=files.bold)
            # degrees_list.append(degrees)

            power_files.append(
                task_utils.write_tsv.with_options(cache_key_fn=None).submit(
                    dataframe=power_connectivity
                )
            )

            task_utils.write_tsv.submit(
                dataframe=degrees,
                filename=(out / f"{utils.img_stem(files.bold)}_degrees").with_suffix(
                    ".tsv"
                ),
            )

            task_utils.write_tsv.submit(
                dataframe=connectivity,
                filename=(
                    out / f"{utils.img_stem(files.bold)}_connectivity"
                ).with_suffix(".tsv"),
            )

            confounds_files.append(
                task_utils.write_tsv.with_options(cache_key_fn=None).submit(
                    dataframe=final_confounds
                )
            )

    task_utils.tsvs_to_parquet.submit(
        acompcor_files, base_dir=out / "task-rest_acompcor"
    )

    task_utils.tsvs_to_parquet.submit(
        confounds_files, base_dir=out / "task-rest_confounds"
    )

    task_utils.tsvs_to_parquet.submit(
        power_files, base_dir=out / "task-rest_atlas-power2011"
    )

    # hub_disruption = get_hub_disruption.submit(degrees_list)
    # task_utils.write_tsv.submit(
    #     dataframe=hub_disruption,
    #     filename=(out / "hubdisruption").with_suffix(".tsv"),
    # )
