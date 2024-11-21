import typing
from pathlib import Path

import polars as pl
import pytest

from biomarkers import datasets


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
    mask_is_file = isinstance(mask, Path)
    mask_exists = mask.exists()
    assert all([mask_is_file, mask_exists])


@pytest.mark.parametrize("weight", [*typing.get_args(datasets.NPSWeights)])
def test_get_nps(weight):
    mask = datasets.get_nps(weight)
    mask_is_file = isinstance(mask, Path)
    mask_exists = mask.exists()
    assert all([mask_is_file, mask_exists])


@pytest.mark.parametrize("weight", [*typing.get_args(datasets.SIIPS1Weights)])
def test_get_siips(weight):
    mask = datasets.get_siips1(weight)
    mask_is_file = isinstance(mask, Path)
    mask_exists = mask.exists()
    assert all([mask_is_file, mask_exists])


@pytest.mark.parametrize("mm", [("2mm"), ("3mm")])
def test_fan_atlas(mm: datasets.FAN_RESOLUTION):
    mask = datasets.get_fan_atlas_file(mm)
    mask_is_file = isinstance(mask, Path)
    mask_exists = mask.exists()
    assert all([mask_is_file, mask_exists])


def test_power():
    coordinates = datasets.get_power2011_coordinates()
    assert isinstance(coordinates, pl.DataFrame)
