from __future__ import annotations
from pathlib import Path

import click

import prefect

from .flows.fslanat import fslanat_flow
from .flows.cat import cat_flow
from .flows.connectivity import connectivity_flow

from ancpbids import BIDSLayout


@prefect.flow
def _main(
    bidslayout: BIDSLayout,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmripreplayout: BIDSLayout | None = None,
    anat: bool = False,
    rest: bool = False,
) -> None:

    if anat:
        fslanat_flow(
            images=set(bidslayout.get(suffix="T1w", extension=".nii.gz")),
            out=output_dir,
        )
        if cat_dir:
            cat_flow(cat_dir=cat_dir, out=output_dir)
    if rest and fmripreplayout:
        connectivity_flow(fmripreplayout=fmripreplayout, out=output_dir)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "bids_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option("--anat", default=False, is_flag=True)
@click.option("--rest", default=False, is_flag=True)
@click.option(
    "--cat-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--fmriprep-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option("--anat", default=False, is_flag=True)
@click.option("--rest", default=False, is_flag=True)
def main(
    bids_dir: Path,
    output_dir: Path = Path("out"),
    cat_dir: Path | None = None,
    fmriprep_dir: Path | None = None,
    anat: bool = False,
    rest: bool = False,
) -> None:

    bidslayout = BIDSLayout(bids_dir)
    fmripreplayout = BIDSLayout(bids_dir) if fmriprep_dir else None

    _main(
        output_dir=output_dir,
        bidslayout=bidslayout,
        fmripreplayout=fmripreplayout,
        anat=anat,
        rest=rest,
        cat_dir=cat_dir,
    )
