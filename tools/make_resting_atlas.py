import re

import pandas as pd
from nilearn import maskers
from scipy import io

mat = io.loadmat("cluster_Fan_Net_r279.mat")

pain_clusters = {0: pd.NA}
for p, pain_cluster in enumerate(
    mat["cluster_Fan_Net"]["pain_cluster_names"][0][0], start=1
):
    pain_clusters.update(
        {p: re.findall(r"[a-zA-Z]+\w*", pain_cluster[0][0])[0]}
    )

full_names = {}
for n, name in enumerate(mat["cluster_Fan_Net"]["full_names"][0][0]):
    full_names.update({n: name[0][0]})


cluster_names = {}
for n, name in enumerate(
    mat["cluster_Fan_Net"]["cluster_names"][0][0], start=1
):
    cluster_names.update({n: name[0][0]})


d = pd.DataFrame(
    mat["cluster_Fan_Net"]["dat"][0][0],
    columns=[
        "original",
        "manual_buckner",
        "manual_bucker2",
        "cluster",
        "pain_cluster",
        "anat_cluster_names",
        "laterality",
        "brainnetome",
        "w_brainstem_cerebellum",
        "update",
        "update2",
        "update3",
        "update9",
    ],
)

d["cluster"] = [cluster_names[x] for x in d["cluster"]]
d["pain_cluster"] = [pain_clusters[x] for x in d["pain_cluster"]]
d["name"] = full_names
d["value"] = range(1, 280)

d[["name", "cluster", "pain_cluster"]].assign(value=range(1, 280)).to_csv(
    "../src/biomarkers/data/fan_atlas.csv", index=False
)

m = maskers.NiftiLabelsMasker(
    labels_img="../src/biomarkers/data/Fan_et_al_atlas_r279_MNI_2mm.nii.gz",
    resampling_target="data",
)
time_series = m.fit_transform(
    imgs="sub-20064_ses-V3_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

import nibabel as nb
import numpy as np

ref = nb.load(
    "/Users/psadil/Downloads/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz"
)
rdlpfc = d.query("pain_cluster == 'dlPFC' and laterality==1").value.tolist()

out = ref.get_fdata().copy()
out[np.logical_not(np.isin(ref.get_fdata(), rdlpfc))] = 0

nb.nifti1.Nifti1Image(out, affine=ref.affine, header=ref.header).to_filename(
    "/Users/psadil/Downloads/data/dlpfc.nii.gz"
)

hipp = d.query("pain_cluster == 'Hipp'").value.tolist()

out2 = ref.get_fdata().copy()
out2[np.logical_not(np.isin(ref.get_fdata(), hipp))] = 0

nb.nifti1.Nifti1Image(out2, affine=ref.affine, header=ref.header).to_filename(
    "/Users/psadil/Downloads/data/hipp.nii.gz"
)
