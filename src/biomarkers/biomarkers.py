from __future__ import annotations
import os
from pathlib import Path

import click

# import prefect
# from prefect.task_runners import SequentialTaskRunner
import prefect_dask

# import prefect_ray
from dask import config


from biomarkers.flows.cat import cat_flow
from biomarkers.flows.connectivity import connectivity_flow
from biomarkers.flows.cuff import cuff_flow

# from .flows.debias import debias_flow
from biomarkers.flows import cuff33


# @prefect.flow(task_runner=SequentialTaskRunner())
def _main(
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    rest_subdirs: frozenset[Path] | None = None,
    cuff_subdirs: frozenset[Path] | None = None,
    fmriprep_subdir: frozenset[Path] | None = None,
) -> None:
    if cat_dir:
        cat_flow.with_options(
            task_runner=prefect_dask.DaskTaskRunner(
                cluster_kwargs={"n_workers": 40, "threads_per_worker": 1}
            )
        )(cat_dir=cat_dir, out=output_dir, return_state=True)
        # cat_flow.with_options(
        #     task_runner=prefect_dask.DaskTaskRunner(
        #         cluster_kwargs={"n_workers": 20, "threads_per_worker": 1}
        #     )
        # )(cat_dir=cat_dir, out=output_dir, return_state=True)
    if rest_subdirs:
        # cuff_flow.with_options(
        #     task_runner=prefect_dask.DaskTaskRunner(
        #         cluster_kwargs={"n_workers": 30, "threads_per_worker": 1}
        #     )
        # )(subdirs=rest_subdirs, out=output_dir, return_state=True)
        connectivity_flow.with_options(
            task_runner=prefect_dask.DaskTaskRunner(
                cluster_kwargs={"n_workers": 20, "threads_per_worker": 1}
            )
        )(subdirs=rest_subdirs, out=output_dir, return_state=True)
    if cuff_subdirs:
        # debias_flow.with_options(
        #     task_runner=prefect_dask.DaskTaskRunner(
        #         cluster_kwargs={"n_workers": 20, "threads_per_worker": 1}
        #     )
        # )(subdirs=cuff_subdirs, out=output_dir, return_state=True)
        cuff_flow.with_options(
            task_runner=prefect_dask.DaskTaskRunner(
                cluster_kwargs={"n_workers": 30, "threads_per_worker": 1}
            )
        )(subdirs=cuff_subdirs, out=output_dir, return_state=True)
        # cuff_flow.with_options(
        #     task_runner=SequentialTaskRunner()
        # )(subdirs=cuff_subdirs, out=output_dir, return_state=True)
    if fmriprep_subdir:
        cuff33.cuff_flow.with_options(
            task_runner=prefect_dask.DaskTaskRunner(
                cluster_kwargs={"n_workers": 20, "threads_per_worker": 1}
            )
        )(subdirs=fmriprep_subdir, out=output_dir, return_state=True)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--cat-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--rest-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--cuff-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--fmriprep-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--tmpdir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
)
@click.option("--sub-limit", type=int, default=None)
def main(
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    rest_dir: Path | None = None,
    cuff_dir: Path | None = None,
    fmriprep_dir: Path | None = None,
    tmpdir: str | None = None,
    sub_limit: int | None = None,
) -> None:
    # this were all used while troubleshooting. It might be worth exploring removing them
    # Though, it's also not clear that there is enough benefit to letting Dask spill memory onto
    # disk, so they are staying off for now
    # the last one would be what to try removing firstish
    config.set(
        {"distributed.worker.memory.rebalance.measure": "managed_in_memory"}
    )
    config.set({"distributed.worker.memory.spill": False})
    config.set({"distributed.worker.memory.target": False})
    config.set({"distributed.worker.memory.pause": False})
    config.set({"distributed.worker.memory.terminate": False})
    config.set({"distributed.comm.timeouts.connect": "90s"})
    config.set({"distributed.comm.timeouts.tcp": "90s"})
    config.set(
        {"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0}
    )

    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
    if not output_dir.exists():
        output_dir.mkdir()

    rest_subdirs = (
        frozenset(list(rest_dir.glob("sub*"))[0:sub_limit])
        if rest_dir
        else None
    )
    cuff_subdirs = (
        frozenset(list(cuff_dir.glob("sub*"))[0:sub_limit])
        if cuff_dir
        else None
    )
    fmriprep_subdirs = (
        frozenset(list(fmriprep_dir.glob("sub*"))[0:sub_limit])
        if fmriprep_dir
        else None
    )
    _main(
        output_dir=output_dir,
        rest_subdirs=rest_subdirs,
        cuff_subdirs=cuff_subdirs,
        cat_dir=cat_dir,
        fmriprep_subdir=fmriprep_subdirs,
    )
