from __future__ import annotations
from pathlib import Path

import click

import prefect

from .flows.fslanat import fslanat_flow
from .flows.cat import cat_flow
from .flows.connectivity import connectivity_flow


@prefect.flow
def _main(
    anats: set(Path) | None = None,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmriprep_dir: Path | None = None,
) -> None:

    if anats:
        fslanat_flow(images=anats, out=output_dir)
    if cat_dir:
        cat_flow(cat_dir=cat_dir, out=output_dir)
    if fmriprep_dir:
        connectivity_flow(fmripreplayout=fmriprep_dir, out=output_dir)


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
        exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
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
def main(
    bids_dir: Path | None = None,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmriprep_dir: Path | None = None,
) -> None:

    _main(
        output_dir=output_dir,
        anats=set(x for x in bids_dir.glob("sub*/**/*T1w.nii.gz")),
        fmriprep_dir=fmriprep_dir,
        cat_dir=cat_dir,
    )
