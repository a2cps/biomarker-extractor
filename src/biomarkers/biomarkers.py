from __future__ import annotations
import os
from pathlib import Path

import click

import prefect

from .flows.fslanat import fslanat_flow
from .flows.cat import cat_flow
from .flows.connectivity import connectivity_flow


@prefect.flow
def _main(
    anats: set[Path] | None = None,
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

    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
    if not output_dir.exists():
        output_dir.mkdir()
    if fmriprep_dir:
        fmriprep_subdirs = frozenset(fmriprep_dir.glob("sub*"))

    anats = set(bids_dir.glob("sub*/ses*/anat/*T1w.nii.gz")) if bids_dir else None
    _main(
        output_dir=output_dir,
        anats=anats,
        fmriprep_subdirs=fmriprep_subdirs,
        cat_dir=cat_dir,
    )
