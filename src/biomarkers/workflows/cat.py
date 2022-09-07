from __future__ import annotations

import nipype

from ..nodes import io
from ..interfaces import catutils

# from niworkflows.interfaces import bids


class CATWF(nipype.Workflow):
    """pull derivatives from CAT12 output

    grabs all available volumes from directory of cat12 outputs

    Args:
        nipype (_type_): _description_
    """

    def __init__(self) -> CATWF:
        super().__init__(name="cat_wf")
        inputnode = io.InputNode.from_fields(["cat_dir"])
        outputnode = io.OutputNode.from_fields(["anat", "volumes"])
        catxml = nipype.Node(catutils.CATXML(), name="catxml")
        self.connect(
            [
                (inputnode, first, [("in_file", "in_file")]),
                (first, volumes, [("anat", "src")]),
                (volumes, outputnode, [("volumes", "volumes")]),
                (first, outputnode, [("anat", "anat")]),
            ]
        )
