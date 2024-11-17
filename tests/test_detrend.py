import numpy as np

from biomarkers import imgs


def test_poly_normal():
    Z = imgs.get_poly_design(10, degree=3)
    ones = np.diag(np.dot(Z.T, Z))
    assert np.isclose(ones, 1).all()


def test_poly_orthogonal():
    Z = imgs.get_poly_design(10, degree=3)
    zeros = np.dot(Z.T, Z)
    np.fill_diagonal(zeros, 0)
    assert np.isclose(zeros, 0).all()


# def test_detrend_preserves_mean():
#     img = nb.nifti1.Nifti1Image.load(
#         "/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep/sub-travel2/ses-NS/func/sub-travel2_ses-NS_task-cuff_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     )
#     mask = Path(
#         "/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep/sub-travel2/ses-NS/func/sub-travel2_ses-NS_task-cuff_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#     )
#     out = imgs.detrend(
#         img=img,
#         mask=mask,
#     )

#     a = masking.apply_mask(img, mask)
#     b = masking.apply_mask(out, mask)

#     assert np.isclose(a.mean(axis=0) - b.mean(axis=0), 0).all()
