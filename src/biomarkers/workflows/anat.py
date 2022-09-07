from __future__ import annotations

from pathlib import Path

import nipype

from .first import FIRSTWF
from .cat import CATWF
from ..nodes import io

# from niworkflows.interfaces import bids

# TODO: allow for re-use of precomputed pipelines (e.g., symlink to .anat dir)


class AnatWF(nipype.Workflow):
    def __init__(self) -> AnatWF:
        super().__init__(name="anat")
        inputnode = io.InputNode.from_fields(["in_file"])
        outputnode = io.OutputNode.from_fields(["anat", "volumes"])
        first_wf = FIRSTWF()
        self.connect(
            [
                (inputnode, first_wf, [("in_file", "inputnode.in_file")]),
                (
                    first_wf,
                    outputnode,
                    [("outputnode.anat", "anat"), ("outputnode.volumes", "volumes")],
                ),
            ]
        )

    @classmethod
    def from_cat(cls, cat_dir: Path) -> AnatWF:
        wf = cls()
        inputnode = io.InputNode.from_fields(
            ["cat_dir"], iterables=[("cat_dir", cat_dir)], name="input_cat"
        )
        outputnode = io.OutputNode.from_fields(["volumes"])

        cat_wf = CATWF()

        wf.connect(
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
                    outputnode,
                    [("outputnode.volumes", "@volumes")],
                ),
            ]
        )
        return wf
