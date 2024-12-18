from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
import pytest
from nilearn import datasets

from biomarkers import imgs
from biomarkers.models import bids

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


def make_mask(img: nb.nifti1.Nifti1Image | Path, tmp_path: Path) -> Path:
    mask = tmp_path / "mask.nii.gz"
    if isinstance(img, Path):
        i = nb.nifti1.load(img)
    else:
        i = img
    nb.nifti1.Nifti1Image(
        np.asarray(i.get_fdata().std(axis=-1, ddof=1) > 0, dtype=np.uint8),
        affine=i.affine,
    ).to_filename(mask)
    return mask


def test_winsorize():
    img: nb.nifti1.Nifti1Image = nb.nifti1.Nifti1Image.load(i)
    n = imgs.winsorize(img=img)
    assert isinstance(n, nb.nifti1.Nifti1Image)


@pytest.mark.filterwarnings("error")
def test_clean_img_no_future_warnings(tmp_path: Path):
    out = tmp_path / "cleaned.nii.gz"
    confounds = tmp_path / "confounds.parquet"
    img = nb.nifti1.Nifti1Image.load(i)
    pl.DataFrame({"x": np.random.normal(size=(img.shape[-1],))}).write_parquet(
        confounds
    )
    mask = make_mask(i, tmp_path)
    # signals = masking.apply_mask(img, mask)
    # signal.clean(
    #     signals,
    #     detrend=False,
    #     standardize="zscore_sample",
    #     standardize_confounds="zscore_sample",  # type: ignore
    #     confounds=pl.read_parquet(confounds).to_numpy(),
    #     low_pass=0.1,
    #     high_pass=0.01,
    #     t_r=utils.get_tr(img),
    # )
    n = imgs.clean_img(
        out,
        img=i,
        mask=mask,
        low_pass=0.1,
        high_pass=0.01,
        confounds_file=confounds,
    )
    assert n.exists()


def test_compcor():
    layout = bids.Layout.from_path(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep")
    )
    filters = {
        "sub": "travel2",
        "space": "MNI152NLin6Asym",
        "res": "2",
    }
    task_filters = {
        "task": "rest",
        "run": "1",
        **filters,
    }
    compcor = imgs.CompCor(
        img=layout.get_preproc(task_filters),
        probseg=bids.ProbSeg.from_layout(layout, filters),
        label="WM+CSF",
        boldref=layout.get_boldref({"suffix": "boldref", **task_filters}),
        high_pass=0.01,
        low_pass=0.1,
        n_non_steady_state_tr=15,
        detrend=True,
    ).fit_transform()

    assert compcor.drop_nulls().shape[0] > 0
