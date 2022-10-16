from pathlib import Path
import tempfile

import pandas as pd
from pyarrow import dataset

import prefect
from prefect.tasks import task_input_hash


def write_tsv(dataframe: pd.DataFrame, filename: Path | None = None) -> Path:
    if filename:
        written = filename
        dataframe.to_csv(filename, index=False, sep="\t")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as f:
            dataframe.to_csv(f, index=False, sep="\t")
            written = Path(f.name)
    return written


@prefect.task
def write_parquet(dataframe: pd.DataFrame, filename: Path | None = None) -> Path:
    dataframe.columns = dataframe.columns.astype(str)
    if filename:
        written = filename
        dataframe.to_parquet(filename, index=False)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            dataframe.to_parquet(f, index=False)
            written = Path(f.name)
    return written


@prefect.task
def combine_parquet(tables: list[Path], base_dir=Path) -> Path:
    d = dataset.dataset(tables, format="parquet")
    dataset.write_dataset(
        data=d,
        base_dir=base_dir,
        format="parquet",
        existing_data_behavior="delete_matching",
    )
    return base_dir
