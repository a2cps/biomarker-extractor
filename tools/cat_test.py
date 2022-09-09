from pathlib import Path
import pandas as pd
from nilearn import maskers

cat = Path(
    "/corral-secure/projects/A2CPS/shared/psadil/products/mris/all_sites/cat/mri"
)
coords = [(2, 52, -2)]
radius = 10

masker = maskers.NiftiSpheresMasker(coords, radius=radius)

inputs = [x for x in cat.glob("mwp1sub*nii")]
time_series = masker.fit_transform(inputs)

d = pd.DataFrame(time_series, columns=coords, index=[x.name for x in inputs])
d.to_csv("vols.tsv", sep="\t", index_label="file")
