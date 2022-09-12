from __future__ import annotations

import nipype

from .first import FIRSTWF
from ..nodes import io

# from niworkflows.interfaces import bids

# TODO: allow for re-use of precomputed pipelines (e.g., symlink to .anat dir)


class AnatWF(nipype.Workflow):
    def __init__(self, **inputs) -> AnatWF:
        super().__init__(name="anat", **inputs)
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
