import typing
from pathlib import Path

import polars as pl
import pytest

from biomarkers import datasets


def check_is_file_exists(src: Path) -> tuple[bool, bool]:
    is_file = isinstance(src, Path)
    exists = src.exists()
    return is_file, exists


@pytest.mark.parametrize(
    "getter",
    [
        (datasets.get_mpfc_mask),
        (datasets.get_power2011_coordinates_file),
        (datasets.get_mni6gray_mask),
    ],
)
def test_get_masks(getter: typing.Callable[..., Path]):
    mask = getter()
    assert all(check_is_file_exists(mask))


@pytest.mark.parametrize("weight", [*typing.get_args(datasets.NPSWeights)])
def test_get_nps(weight):
    mask = datasets.get_nps(weight)
    assert all(check_is_file_exists(mask))


@pytest.mark.parametrize("weight", [*typing.get_args(datasets.SIIPS1Weights)])
def test_get_siips(weight):
    mask = datasets.get_siips1(weight)
    assert all(check_is_file_exists(mask))


@pytest.mark.parametrize("mm", [(2), (3)])
def test_fan_atlas(mm: datasets.FanResolution):
    mask = datasets.get_fan_atlas_nii_file(mm)
    assert all(check_is_file_exists(mask))


def test_power():
    coordinates = datasets.get_power2011_coordinates()
    assert isinstance(coordinates, pl.DataFrame)


def test_cat12batch():
    batch = datasets.get_cat_batch()
    assert all(check_is_file_exists(batch))
