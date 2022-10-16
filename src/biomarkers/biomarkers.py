from __future__ import annotations
import os
from pathlib import Path

import click

import prefect
from dask import config

from .flows.fslanat import fslanat_flow
from .flows.cat import cat_flow
from .flows.connectivity import connectivity_flow


@prefect.flow
def _main(
    anats: frozenset[Path] | None = None,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmriprep_subdirs: frozenset[Path] | None = None,
) -> None:

    if anats:
        fslanat_flow(images=anats, out=output_dir)
    if cat_dir:
        cat_flow(cat_dir=cat_dir, out=output_dir)
    if fmriprep_subdirs:
        connectivity_flow(subdirs=fmriprep_subdirs, out=output_dir)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--bids-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--cat-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--fmriprep-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--tmpdir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
def main(
    bids_dir: Path | None = None,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmriprep_dir: Path | None = None,
    tmpdir: str | None = None,
) -> None:

    config.set({"distributed.worker.memory.rebalance.measure": "managed_in_memory"})
    config.set({"distributed.worker.memory.spill": False})
    config.set({"distributed.worker.memory.target": False})
    config.set({"distributed.worker.memory.pause": False})
    config.set({"distributed.worker.memory.terminate": False})
    config.set({"distributed.comm.timeouts.connect": "90s"})
    config.set({"distributed.comm.timeouts.tcp": "90s"})
    config.set({"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0})

    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
    if not output_dir.exists():
        output_dir.mkdir()
    if fmriprep_dir:
        fmriprep_subdirs = frozenset(fmriprep_dir.glob("sub*"))

    anats = frozenset(bids_dir.glob("sub*/ses*/anat/*T1w.nii.gz")) if bids_dir else None
    _main(
        output_dir=output_dir,
        anats=anats,
        fmriprep_subdirs=fmriprep_subdirs,
        cat_dir=cat_dir,
    )
