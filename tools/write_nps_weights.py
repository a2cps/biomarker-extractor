import nibabel as nb


def pair_to_img(x: nb.Nifti1Pair) -> nb.Nifti1Image:
    return nb.Nifti1Image(dataobj=x.dataobj, affine=x.affine, header=x.header)


x = nb.load("2013_Wager_NEJM_NPS/weights_NSF_grouppred_cvpcr.img")

y = pair_to_img(x)

y.to_filename("../src/biomarkers/data/weights_NSF_grouppred_cvpcr.nii.gz")

x = nb.load("2013_Wager_NEJM_NPS/weights_NSF_positive_smoothed_larger_than_10vox.img")
y = pair_to_img(x)
y.to_filename(
    "../src/biomarkers/data/weights_NSF_positive_smoothed_larger_than_10vox.nii.gz"
)

x = nb.load("2013_Wager_NEJM_NPS/weights_NSF_negative_smoothed_larger_than_10vox.img")
y = pair_to_img(x)
y.to_filename(
    "../src/biomarkers/data/weights_NSF_negative_smoothed_larger_than_10vox.nii.gz"
)

# strange. the "positive" file has negative weights? probably okay
x0 = nb.load("2013_Wager_NEJM_NPS/weights_NSF_positive_smoothed_larger_than_10vox.img")
x1 = nb.load("2013_Wager_NEJM_NPS/weights_NSF_negative_smoothed_larger_than_10vox.img")

y = nb.Nifti1Image(
    dataobj=x0.get_fdata() + x1.get_fdata(), affine=x.affine, header=x.header
)
y.to_filename("../src/biomarkers/data/weights_NSF_smoothed_larger_than_10vox.nii.gz")
