import logging
from pathlib import Path

import nibabel as nb
import polars as pl
from nilearn import maskers

from biomarkers import datasets, utils


def conform_orientation(f: Path) -> Path:
    nii = nb.nifti1.Nifti1Image.load(f)
    input_axcodes = nb.orientations.aff2axcodes(nii.affine)
    input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
    desired_orientation = nb.orientations.axcodes2ornt(("LAS"))
    transform_orientation = nb.orientations.ornt_transform(
        input_orientation, desired_orientation
    )
    nii.as_reoriented(transform_orientation).to_filename(f)  # type:ignore
    return f


async def transform_jhu_labels(transform: Path, reference: Path, dst: Path) -> Path:
    logging.info("Transforming labels to native space")
    utils.mkdir_recursive(dst.parent)
    async with utils.subprocess_manager(
        log=Path("/dev/null"),
        args=[
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(datasets.get_jhu_dti()),
            "--interpolation",
            "GenericLabel",
            "-t",
            str(transform),
            "-r",
            str(reference),
            "-o",
            str(dst),
        ],
    ) as proc:
        await proc.wait()

    logging.info("Conforming orientation of transformed labels")
    return conform_orientation(dst)


async def postdtifit_flow(
    dtifit: Path, qsiprep: Path, outdir: Path, sub: str, ses: str
) -> None:
    labels_img = await transform_jhu_labels(
        transform=qsiprep
        / f"sub-{sub}"
        / "anat"
        / f"sub-{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        reference=qsiprep
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "dwi"
        / f"sub-{sub}_ses-{ses}_space-T1w_dwiref.nii.gz",
        dst=outdir
        / "dtifit_regional"
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "dwi"
        / f"sub-{sub}_ses-{ses}_space-dwifslstd_desc-JHUICBM_dseg.nii.gz",
    )

    logging.info("Extracting DTI metrics")
    jhu_dti = datasets.get_jhu_lut()
    out: list[pl.DataFrame] = []
    for stat in ["mean", "minimum", "maximum"]:
        masker = maskers.NiftiLabelsMasker(labels_img=labels_img, strategy=stat)
        for metric in ["FA", "MD", "MO", "L1", "L2", "L3"]:
            for shells in ["b1000", "multishell"]:
                dir_dtifit = dtifit / shells / f"sub-{sub}" / f"ses-{ses}" / "dwi"
                if shells == "multishell":
                    dti = (
                        dir_dtifit
                        / f"sub-{sub}_ses-{ses}_space-T1w_dwi_{metric}.nii.gz"
                    )
                else:
                    dti = (
                        dir_dtifit
                        / f"sub-{sub}_ses-{ses}_acq-{shells}_space-T1w_dwi_{metric}.nii.gz"
                    )
                out.append(
                    pl.DataFrame({"value": masker.fit_transform(dti).squeeze()})
                    .with_row_index("index", offset=1)
                    .with_columns(
                        pl.col("index").cast(pl.UInt8),
                        stat=pl.lit(stat),
                        metric=pl.lit(metric),
                        sub=int(sub),
                        ses=pl.lit(ses),
                        shells=pl.lit(shells),
                    )
                    .join(jhu_dti, on=["index"])
                )

    logging.info("Saving dti metrics")
    pl.concat(out).pivot(on="stat", values="value").lazy().sink_parquet(
        pl.PartitionByKey(
            outdir / "dtifit_regional_stats", by=[pl.col.sub, pl.col.ses]
        ),
        mkdir=True,
    )
