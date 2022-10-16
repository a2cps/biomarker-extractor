from __future__ import annotations
from pathlib import Path

import prefect
from prefect.tasks import task_input_hash
from prefect_dask import DaskTaskRunner

from ..models.cat import CATResult
from .. import utils


@prefect.task(cache_key_fn=task_input_hash)
def write_cat_volumes(catresult: CATResult, out: Path, suffix=str) -> None:
    catresult.write_volumes(filename=out / f"{utils.img_stem(catresult.img)}{suffix}")


@prefect.task(cache_key_fn=task_input_hash)
def get_catresult(root: Path, img: Path) -> CATResult:
    return CATResult.from_root(root=root, img=img)


@prefect.task
def _cat(image: Path, cat_dir: Path, out: Path) -> Path:
    filename = out / f"{utils.img_stem(image)}_mpfc.tsv"
    if filename.exists():
        print(f"Found existing catresult output {filename}. Not running")
    else:
        catresult = CATResult.from_root(root=cat_dir, img=image)
        catresult.write_volumes(filename=filename)
    return filename


@prefect.flow(task_runner=DaskTaskRunner)
def cat_flow(cat_dir: Path, out: Path) -> None:
    for anat in frozenset(cat_dir.glob("*.nii.gz")):
        _cat.submit(anat, cat_dir=cat_dir, out=out)
