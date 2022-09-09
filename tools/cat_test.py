from pathlib import Path
import pandas as pd
from nilearn import maskers
import numpy as np

cat = Path(
    "/corral-secure/projects/A2CPS/shared/psadil/products/mris/all_sites/cat/mri"
)
coords = [(2, 52, -2)]
radius = 10

masker = maskers.NiftiSpheresMasker(coords, radius=radius)

inputs = [x for x in cat.glob("mwp1sub*nii")]
time_series = masker.fit_transform(inputs)
volumes = time_series * 4.0 / 3.0 * np.pi * radius**3

d = pd.DataFrame(
    {coords[0]: time_series[0], "volume": volumes[0]},
    index=[x.name for x in inputs],
)
d.to_csv("vols.tsv", sep="\t", index_label="file")
