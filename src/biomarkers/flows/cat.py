from pathlib import Path

import prefect
from prefect_dask import DaskTaskRunner

from ..models.cat import CATResult
from .. import utils


@prefect.task
def _cat(image: Path, cat_dir: Path, out: Path) -> Path:
    filename = out / f"{utils.img_stem(image)}_mpfc.tsv"
    if filename.exists():
        print(f"Found existing catresult output {filename}. Not running")
    else:
        catresult = CATResult.from_root(root=cat_dir, img=image)
        catresult.write_volumes(filename=filename)
    return filename


@prefect.flow
def cat_flow(cat_dir: Path, out: Path) -> None:
    _cat.map(cat_dir.glob("*.nii.gz"), cat_dir=cat_dir, out=out)
