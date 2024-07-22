from importlib import resources
from pathlib import Path


def get_cat_batch() -> Path:
    with resources.path("biomarkers.data", "batch.m") as f:
        mpfc = f
    return mpfc
