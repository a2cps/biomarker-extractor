from __future__ import annotations

import pathlib

from nipype.interfaces.base import (
    TraitedSpec,
    ImageFile,
    Directory,
    SimpleInterface,
)


class CATXMLInputSpec(TraitedSpec):
    feat_dir = Directory(
        exists=True,
        manditory=True,
        resolve=True,
        desc="directory of FEAT analysis",
    )


class CATXMLOutputSpec(TraitedSpec):
    filtered_func_data = ImageFile(desc="image file from input")
    example_func2standard_warp = ImageFile(desc="for use with applywarp")


class CATXML(SimpleInterface):

    input_spec = CATXMLInputSpec
    output_spec = CATXMLOutputSpec

    def _run_interface(self, runtime):
        self._results["filtered_func_data"] = (
            pathlib.Path(self.inputs.feat_dir) / "filtered_func_data.nii.gz"
        )
        self._results["example_func2standard_warp"] = (
            pathlib.Path(self.inputs.feat_dir)
            / "reg"
            / "example_func2standard_warp.nii.gz"
        )
        return runtime
