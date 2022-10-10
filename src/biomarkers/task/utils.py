from pathlib import Path
import tempfile

import pandas as pd
from pyarrow import dataset

import prefect
from prefect.tasks import task_input_hash


@prefect.task(cache_key_fn=task_input_hash)
def write_tsv(dataframe: pd.DataFrame, filename: Path | None) -> Path:
    if filename:
        written = filename
        dataframe.to_csv(filename, index=False, sep="\t")
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            dataframe.to_csv(f, index=False, sep="\t")
            written = Path(f.name)
    return written


@prefect.task
def tsvs_to_parquet(tables: list[Path], base_dir=Path) -> Path:
    d = dataset.dataset(tables)
    dataset.write_dataset(data=d, base_dir=base_dir, format="parquet")
    return base_dir
