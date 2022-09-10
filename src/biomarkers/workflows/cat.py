from __future__ import annotations

import nipype

from ..nodes import io
from ..interfaces import catutils


class CATWF(nipype.Workflow):
    """pull derivatives from CAT12 output

    grabs all available volumes from directory of cat12 outputs

    Args:
        nipype (_type_): _description_
    """

    def __init__(self, **inputs) -> CATWF:
        super().__init__(name="cat_wf", **inputs)
        inputnode = io.InputNode.from_fields(["cat_dir"])
        outputnode = io.OutputNode.from_fields(["volumes"])

        volumes = nipype.Node(catutils.CATVol(), name="volumes")
        volumes.inputs.glob = "mwp1sub*nii"

        self.connect(
            [
                (inputnode, volumes, [("cat_dir", "cat_dir")]),
                (volumes, outputnode, [("volumes", "volumes")]),
            ]
        )
