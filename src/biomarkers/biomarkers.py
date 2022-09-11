from __future__ import annotations
from pathlib import Path

import click

import nipype
import bids

from .nodes import io
from .workflows.anat import AnatWF
from .workflows.rest import RestWF
from .workflows.cat import CATWF


class MainWF(nipype.Workflow):
    def __init__(self, **kwargs):
        super().__init__(name="biomarkers", **kwargs)

    @classmethod
    def from_layout(
        cls,
        layout: bids.layout.BIDSLayout,
        output_dir: Path,
        anat: bool = True,
        rest: bool = True,
        cat_dir: Path | None = None,
        **kwargs,
    ) -> MainWF:
        wf = cls(**kwargs)
        datasink = nipype.Node(
            nipype.DataSink(
                base_directory=output_dir,
                regexp_substitutions=[
                    (r"_in_file_.*/", ""),
                ],
            ),
            name="datasink",
            run_without_submitting=True,
        )
        if anat:
            t1w = layout.get(return_type="file", suffix="T1w", extension="nii.gz")
            wf._connect_anat(anat=t1w, datasink=datasink)
        if rest:
            t1w = layout.get(return_type="file", suffix="T1w", extension="nii.gz")
            bold = layout.get(
                return_type="file", suffix="bold", extension="nii.gz", task="rest"
            )
            wf._connect_rest(bold, anat=t1w[0], datasink=datasink)
        if cat_dir:
            wf._connect_cat(cat_dir)
        return wf

    def _connect_anat(
        self,
        anat: list[Path],
        datasink: nipype.Node,
    ) -> MainWF:
        inputnode = io.InputNode.from_fields(
            ["in_file"], iterables=[("in_file", anat)], name="input_anat"
        )
        wf = AnatWF()

        self.connect(
            [
                (inputnode, wf, [("in_file", "inputnode.in_file")]),
                (
                    wf,
                    datasink,
                    [
                        ("outputnode.volumes", "@volumes"),
                        ("outputnode.anat", "@anat"),
                    ],
                ),
            ]
        )
        return self

    def _connect_rest(
        self, rest: list[Path], anat: Path, datasink: nipype.Node
    ) -> MainWF:
        inputnode = io.InputNode.from_fields(
            ["in_file", "anat"], iterables=[("in_file", rest)], name="input_rest"
        )
        inputnode.inputs.anat = anat
        wf = RestWF()
        self.connect(
            [
                (
                    inputnode,
                    wf,
                    [
                        ("in_file", "inputnode.in_file"),
                        ("anat", "inputnode.anat"),
                    ],
                ),
                (
                    wf,
                    datasink,
                    [("outputnode.correlation_matrix", "@correlation_matrix")],
                ),
            ]
        )
        return self

    def _connect_cat(self, cat_dir: Path, datasink: nipype.Node) -> MainWF:
        inputnode = io.InputNode.from_fields(["cat_dir"], name="input_cat")
        inputnode.inputs.cat_dir = cat_dir
        cat_wf = CATWF()

        self.connect(
            [
                (
                    inputnode,
                    cat_wf,
                    [
                        ("cat_dir", "inputnode.cat_dir"),
                    ],
                ),
                (
                    cat_wf,
                    datasink,
                    [("outputnode.volumes", "@catvolumes")],
                ),
            ]
        )

        pass


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "src",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--base-dir",
    default="work",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--layout-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--cat-dir",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option("--anat", default=False, is_flag=True)
@click.option("--rest", default=False, is_flag=True)
@click.option(
    "--plugin",
    default="Linear",
    type=click.Choice(choices=["Linear", "MultiProc"]),
)
def main(
    src: Path,
    output_dir: Path = Path("out"),
    base_dir: Path = Path("work"),
    cat_dir: Path | None = None,
    anat: bool = False,
    rest: bool = False,
    plugin: str = "Linear",
    layout_dir: Path | None = None,
) -> None:

    if layout_dir:
        layout = bids.layout.BIDSLayout(database_path=layout_dir)
    else:
        layout = bids.BIDSLayout(root=src)

    wf = MainWF.from_layout(
        output_dir=output_dir,
        base_dir=base_dir,
        layout=layout,
        anat=anat,
        rest=rest,
        cat_dir=cat_dir,
    )

    wf.run(plugin)
