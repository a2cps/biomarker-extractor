import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
import pydantic
from nilearn import maskers
from nilearn.connectome import ConnectivityMeasure
from sklearn import covariance

from biomarkers import datasets, utils
from biomarkers.models import functional_connectivity as fnc_models
from biomarkers.models import postprocess

YEO_NETWORKS: tuple[datasets.YeoNetworks, ...] = typing.get_args(datasets.YeoNetworks)
SCHAEFER_N_ROIS: tuple[datasets.SchaeferNROI, ...] = typing.get_args(
    datasets.SchaeferNROI
)
SCHAEFERR_RESOLUTIONS: tuple[datasets.SchaeferResolution, ...] = typing.get_args(
    datasets.SchaeferResolution
)
FAN_RESOLUTIONS: tuple[datasets.FanResolution, ...] = (2,)
DIFUMO_DIMENSIONS: tuple[datasets.DIFUMODimension, ...] = (64, 1024)
DIFUMO_RESOLUTIONS: tuple[datasets.DIFUMOResolution, ...] = (2,)
GORDON_RESOLUTIONS: tuple[datasets.GordonResolution, ...] = (2,)
GORDON_SPACES: tuple[datasets.GordonSpace, ...] = ("MNI",)


class Coordinate(typing.NamedTuple):
    x: int
    y: int
    z: int


def _get_labels() -> dict[str, datasets.Labels]:
    out = {}
    for n in fnc_models.SCHAEFER_N_ROIS:
        for networks in fnc_models.YEO_NETWORKS:
            out.update(
                {
                    f"schaefer_nrois-{n}_resolution-2_networks-{networks}": datasets.get_atlas_schaefer_2018(
                        n_rois=n, resolution_mm=2, yeo_networks=networks
                    )
                }
            )
    for resolution in fnc_models.FAN_RESOLUTIONS:
        out.update(
            {
                f"fan_resolution-{resolution}": datasets.get_fan_atlas(
                    resolution=resolution
                )
            }
        )
    out.update(
        {
            "gordon_space-mni_resolution-2": datasets.get_atlas_gordon_2016(
                resolution_mm=2, space="MNI"
            )
        }
    )
    return out


def _get_maps() -> dict[str, datasets.Labels]:
    out = {}
    for dimension in fnc_models.DIFUMO_DIMENSIONS:
        for mm in fnc_models.DIFUMO_RESOLUTIONS:
            out.update(
                {
                    f"difumo_dimension-{dimension}_resolution-{mm}mm": datasets.get_difumo(
                        dimension=dimension, resolution_mm=mm
                    )  # type: ignore
                }
            )
    return out


def _get_coordinates() -> dict[str, dict[int, fnc_models.Coordinate]]:
    return {"dmn": get_baliki_coordinates(), "power": get_power_coordinates()}


def _get_estimators() -> dict[str, type[covariance.EmpiricalCovariance]]:
    return {
        "empirical": covariance.EmpiricalCovariance,
        "leodit_wolf": covariance.LedoitWolf,
        "minimum_covariance_determinant": covariance.MinCovDet,
    }


class PostProcessRunFlow(pydantic.BaseModel):
    process_flow: postprocess.PostProcessRunFlow
    estimators: dict[str, type[covariance.EmpiricalCovariance]] = pydantic.Field(
        default_factory=_get_estimators
    )
    coordinates: dict[str, dict[int, fnc_models.Coordinate]] = pydantic.Field(
        default_factory=_get_coordinates
    )
    maps: dict[str, datasets.Labels] = pydantic.Field(default_factory=_get_maps)
    labels: dict[str, datasets.Labels] = pydantic.Field(default_factory=_get_labels)

    def run(self):
        self.process_flow.clean()

        with tempfile.TemporaryDirectory() as tmpd_:
            tmpd = Path(tmpd_) / "connectivity"
            space_d = (
                tmpd
                / f"sub={self.process_flow.sub}"
                / f"ses={self.process_flow.ses}"
                / f"task={self.process_flow.task}"
                / f"run={self.process_flow.run}"
                / f"space={self.process_flow.space}"
            )

            for e, estimator in self.estimators.items():
                for atlas, label in self.labels.items():
                    get_labels_connectivity(
                        space_d
                        / f"atlas={atlas}"
                        / f"estimator={e}"
                        / "part-0.parquet",
                        img=self.process_flow.cleaned,
                        labels=label,
                        estimator=estimator,
                        mask_img=self.process_flow.mask,
                    )

                for atlas, m in self.maps.items():
                    get_maps_connectivity(
                        space_d
                        / f"atlas={atlas}"
                        / f"estimator={e}"
                        / "part-0.parquet",
                        img=self.process_flow.cleaned,
                        maps=m,
                        estimator=estimator,
                        mask_img=self.process_flow.mask,
                    )

                for key, value in self.coordinates.items():
                    get_coordinates_connectivity(
                        space_d / f"atlas={key}" / f"estimator={e}" / "part-0.parquet",
                        img=self.process_flow.cleaned,
                        coordinates=value,
                        estimator=estimator,
                    )

            utils.write_parquet(
                pl.read_parquet(tmpd),
                self.process_flow.dst
                / "connectivity"
                / f"sub={self.process_flow.sub}"
                / f"ses={self.process_flow.ses}"
                / f"task={self.process_flow.task}"
                / "part-0.parquet",
            )


def df_to_coordinates(df: pl.DataFrame) -> dict[int, fnc_models.Coordinate]:
    coordinates: dict[int, fnc_models.Coordinate] = {}
    for row in df.iter_rows(named=True):
        coordinates.update(
            {row["region"]: fnc_models.Coordinate(x=row["x"], y=row["y"], z=row["z"])}
        )

    return coordinates


def get_baliki_coordinates() -> dict[int, fnc_models.Coordinate]:
    coordinates = datasets.get_baliki_lut()
    return df_to_coordinates(coordinates)


def get_power_coordinates() -> dict[int, fnc_models.Coordinate]:
    coordinates = datasets.get_coordinates_power2011()
    return df_to_coordinates(coordinates)


def do_connectivity(
    time_series: np.ndarray,
    labels: typing.Sequence[int],
    estimator: type[covariance.EmpiricalCovariance],
) -> pl.DataFrame:
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=estimator(store_precision=False),  # type: ignore
        kind="correlation",
        standardize="zscore_sample",  # type: ignore
    )
    correlation_matrix = connectivity_measure.fit_transform([time_series])
    if not isinstance(correlation_matrix, np.ndarray):
        msg = f"Unexpected {type(correlation_matrix)=}"
        raise RuntimeError(msg)

    # need squeeze because the first axis corresponds to subs, which is
    # in this case always length 1. We get a 2d array.
    return utils.mat_to_df(correlation_matrix.squeeze(), labels=labels)


@utils.cache_dataframe
def get_coordinates_connectivity(
    img: Path,
    coordinates: dict[int, fnc_models.Coordinate],
    estimator: type[covariance.EmpiricalCovariance],
    radius: int = 5,  # " ... defined as 10-mm spheres centered ..."
) -> pl.DataFrame:
    # we do not include a mask because many of the images
    # had only a partial FOV, and NiftiSpheresMasker raises
    # an error when including a seed that is outside the mask
    masker = maskers.NiftiSpheresMasker(
        seeds=coordinates.values(), radius=radius, standardize=False, detrend=False
    )
    time_series = masker.fit_transform(img)
    if not isinstance(time_series, np.ndarray):
        msg = f"Unexpected {type(time_series)=}"
        raise RuntimeError(msg)

    return do_connectivity(
        time_series=time_series, labels=tuple(coordinates.keys()), estimator=estimator
    )


@utils.cache_dataframe
def get_maps_connectivity(
    img: Path,
    maps: datasets.Labels,
    estimator: type[covariance.EmpiricalCovariance],
    mask_img: Path | None = None,
) -> pl.DataFrame:
    masker = maskers.NiftiMapsMasker(
        maps_img=maps.labels_img,
        standardize=False,
        detrend=False,
        resampling_target="data",
        mask_img=mask_img,
    )
    time_series = masker.fit_transform(img)
    if not isinstance(time_series, np.ndarray):
        msg = f"Unexpected {type(time_series)=}"
        raise RuntimeError(msg)

    return do_connectivity(
        time_series=time_series,
        labels=maps.labels.select("region").to_series().to_list(),
        estimator=estimator,
    )


@utils.cache_dataframe
def get_labels_connectivity(
    img: Path,
    labels: datasets.Labels,
    estimator: type[covariance.EmpiricalCovariance],
    mask_img: Path | None = None,
) -> pl.DataFrame:
    masker = maskers.NiftiLabelsMasker(
        labels_img=labels.labels_img,
        standardize=False,
        detrend=False,
        resampling_target="data",
        mask_img=mask_img,
    )
    # need to fit here in case of loss of labels
    time_series = masker.fit_transform(img)
    labels_lookup = _update_labels(masker._resampled_labels_img_, labels.labels)  # type: ignore

    if not isinstance(time_series, np.ndarray):
        msg = f"Unexpected {type(time_series)=}"
        raise RuntimeError(msg)

    return do_connectivity(
        time_series=time_series,
        labels=labels_lookup["region"].to_list(),
        estimator=estimator,
    )


def _update_labels(
    resampled_labels_img: nb.nifti1.Nifti1Image, labels: pl.DataFrame
) -> pl.DataFrame:
    resampled_labels = np.unique(np.asarray(resampled_labels_img.dataobj, dtype=int))
    out = labels.filter(pl.col("region").is_in(resampled_labels))
    # - 1 is for background value
    if not (len(resampled_labels) - 1) == out.shape[0]:
        msg = "we appear to have lost labels. this should not be possible."
        raise AssertionError(msg)

    return out
