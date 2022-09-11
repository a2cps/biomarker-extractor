from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from nilearn import maskers

from nipype.interfaces.base import (
    TraitedSpec,
    Directory,
    SimpleInterface,
    Str,
    File,
)


class CATVolInputSpec(TraitedSpec):
    cat_dir = Directory(
        exists=True,
        manditory=True,
        resolve=True,
        desc="directory containing all cat12 outputs",
    )
    glob = Str(manditory=True, desc="glob pattern to match files in cat_dir")


class CATVolOutputSpec(TraitedSpec):
    volumes = File(desc="tsv file containing volumes")


class CATVol(SimpleInterface):

    input_spec = CATVolInputSpec
    output_spec = CATVolOutputSpec

    def __init__(self, **inputs):
        super().__init__(**inputs)

    def _run_interface(self, runtime):
        coords = [(2, 52, -2)]
        radius = 10

        masker = maskers.NiftiSpheresMasker(coords, radius=radius)

        inputs = [x for x in pathlib.Path(self.inputs.cat_dir).glob(self.inputs.glob)]
        time_series = masker.fit_transform(inputs)
        volumes = time_series * 4.0 / 3.0 * np.pi * radius**3

        d = pd.DataFrame(
            {coords[0]: time_series[:, 0], "volume": volumes[:, 0]},
            index=[x.name for x in inputs],
        )
        self._results["volumes"] = str(pathlib.Path("mpfc-gm.tsv").absolute())
        d.to_csv(self._results["volumes"], sep="\t", index_label="file")

        return runtime
