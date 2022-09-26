from pathlib import Path

import pandas as pd

import prefect
from prefect.tasks import task_input_hash


@prefect.task(cache_key_fn=task_input_hash)
# @prefect.task
def write_tsv(dataframe: pd.DataFrame, filename: Path) -> None:
    dataframe.to_csv(filename, index=False, sep="\t")


@prefect.task
def _get(layout, layoutargs: dict):
    return layout.get(**layoutargs)


@prefect.task
def _get_subjects(layout, layoutargs: dict | None = None) -> list[str]:
    if layoutargs is None:
        return layout.get_subjects()
    else:
        return layout.get_subjects(**layoutargs)


@prefect.task
def _get_sessions(layout, layoutargs: dict | None = None) -> list[str]:
    if layoutargs is None:
        return layout.get_sessions()
    else:
        return layout.get_sessions(**layoutargs)


@prefect.task
def _get_runs(layout, layoutargs: dict | None = None) -> list[str]:
    if layoutargs is None:
        return layout.get_runs()
    else:
        return layout.get_runs(**layoutargs)
