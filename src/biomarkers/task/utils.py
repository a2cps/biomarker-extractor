from pathlib import Path

import pandas as pd

import prefect
from prefect.tasks import task_input_hash


@prefect.task(cache_key_fn=task_input_hash)
# @prefect.task
def write_tsv(dataframe: pd.DataFrame, filename: Path) -> None:
    dataframe.to_csv(filename, index=False, sep="\t")
