from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nilearn import datasets

from biomarkers import imgs

_, urls = datasets.fetch_ds000030_urls()

exclusion_patterns = [
    "*group*",
    "*phenotype*",
    "*mriqc*",
    "*parameter_plots*",
    "*physio_plots*",
    "*space-fsaverage*",
    "*space-T1w*",
    "*dwi*",
    "*beh*",
    "*task-bart*",
    "*task-rest*",
    "*task-scap*",
    "*task-task*",
    "*derivatives*",
    "*anat*",
]
urls = datasets.select_from_index(
    urls, exclusion_filters=exclusion_patterns, n_subjects=1
)

data_dir, _ = datasets.fetch_openneuro_dataset(urls=urls)

i = Path(data_dir) / "sub-10159" / "func" / "sub-10159_task-stopsignal_bold.nii.gz"


def test_winsorize_wo_mask_warns():
    with pytest.warns(RuntimeWarning):
        imgs.winsorize(nb.nifti1.load(i))  # type: ignore


def test_winsorize(tmp_path: Path):
    mask = tmp_path / "mask.nii.gz"
    img: nb.nifti1.Nifti1Image = nb.nifti1.load(i)  # type:ignore
    nb.nifti1.Nifti1Image(
        np.asarray(
            img.get_fdata().std(axis=-1, ddof=1, keepdims=True) > 0, dtype=np.uint8
        ),
        affine=img.affine,
    ).to_filename(mask)

    imgs.winsorize(img=img, mask=mask)
