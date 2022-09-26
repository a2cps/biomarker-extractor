from __future__ import annotations
from pathlib import Path

import prefect
from prefect.tasks import task_input_hash
from prefect_dask import DaskTaskRunner

from ..models.cat import CATResult
from .. import utils


@prefect.task(cache_key_fn=task_input_hash)
def write_cat_volumes(catresult: CATResult, filename: Path) -> None:
    catresult.write_volumes(filename=filename)


@prefect.flow(task_runner=DaskTaskRunner)
def cat_flow(cat_dir: Path, out: Path) -> None:

    for i in cat_dir.glob("*.nii.gz"):
        # create all results
        catresult = CATResult.from_root(root=cat_dir, img=i)
        write_cat_volumes.submit(
            catresult, out / f"{utils._img_basename(catresult.img)}_mpfc.tsv"
        )
