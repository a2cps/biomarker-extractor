from pathlib import Path

import pandas as pd
from nilearn import datasets as nilearn_datasets

data_dir = Path("../src/biomarkers/data")

for dimension in [64, 128, 256, 512, 1024]:
    for mm in [2, 3]:
        nilearn_datasets.fetch_atlas_difumo(
            dimension=dimension,
            resolution_mm=mm,
            data_dir=data_dir,
            legacy_format=False,
        )

for n in [400]:
    for networks in [7, 17]:
        nilearn_datasets.fetch_atlas_schaefer_2018(
            n_rois=n, resolution_mm=2, yeo_networks=networks, data_dir=data_dir
        )


rois: pd.DataFrame = nilearn_datasets.fetch_coords_power_2011(
    legacy_format=False
).rois
rois.rename(columns={"roi": "region"}).to_csv(
    data_dir / "power2011_atlas.tsv", index=False, sep="\t"
)

# gordon parcels downloaded from (based on https://doi.org/10.1093/cercor/bhu239)
# https://sites.wustl.edu/petersenschlaggarlab/files/2018/06/Parcels-19cwpgu.zip