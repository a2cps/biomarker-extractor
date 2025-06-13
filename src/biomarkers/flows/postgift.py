import gc
import logging
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
from nilearn import maskers, regions
from scipy import io

from biomarkers import datasets, utils

DMN_COMPONENT = 91 - 1
SLN_COMPONENT = 95 - 1
INSULA = (-42, -2, 2)


def reconstruct_nifti(timecourse: Path) -> nb.nifti1.Nifti1Image:
    nii: nb.nifti1.Nifti1Image = regions.signals_to_img_maps(
        region_signals=nb.nifti1.Nifti1Image.load(timecourse).get_fdata(),
        maps_img=timecourse.with_name(
            timecourse.name.replace("timecourses", "ic_maps")
        ),
    )  # type: ignore
    return nii


def extract_component_map_connectivity(
    src: Path, nii: nb.nifti1.Nifti1Image, component: int, map: Path
) -> np.float64:
    masker = maskers.NiftiMapsMasker(map)
    s1 = masker.fit_transform(nii).squeeze()
    dmn = nb.nifti1.Nifti1Image.load(src).get_fdata()[:, component]
    return np.corrcoef(s1, dmn, dtype=np.float64)[0, 1]


def extract_dmn_insula(
    src: Path, nii: nb.nifti1.Nifti1Image, radius: float = 5
) -> np.float64:
    # Product-moment correlation between DMN network component and 5mm seed around in left Insula (-42, -2, 2)
    # nii = reconstruct_nifti(src)
    insula_masker = maskers.NiftiSpheresMasker([INSULA], radius=radius)
    insula = insula_masker.fit_transform(nii).squeeze()
    dmn = nb.nifti1.Nifti1Image.load(src).get_fdata()[:, DMN_COMPONENT]
    return np.corrcoef(insula, dmn, dtype=np.float64)[0, 1]


def model_from_path(f: Path) -> float:
    return 2.1 if "neuromark_fmri_2.1" in str(f) else 2.0


def extract_fcn_corrs(f: Path) -> pl.DataFrame:
    postprocess = io.loadmat(f, squeeze_me=True, variable_names=["fnc_corrs"])
    return (
        pl.DataFrame(postprocess["fnc_corrs"])
        .with_columns(source=pl.int_range(pl.len(), dtype=pl.UInt16))
        .unpivot(index="source", variable_name="target", value_name="connectivity")
        .with_columns(pl.col("target").str.strip_prefix("column_").cast(pl.UInt16))
        .filter(pl.col("source") < pl.col("target"))
        .with_columns(
            sub=pl.lit(utils.get_sub(f)),
            ses=pl.lit(utils.get_ses(f)),
            task=pl.lit(utils.get_entity(f, "(?<=task-)[a-z]+")),
            run=int(utils.get_entity(f, r"(?<=run-)\d+")),
            model=model_from_path(f),
        )
    )


def extract_falff(f: Path) -> pl.DataFrame:
    postprocess = io.loadmat(f, squeeze_me=True, variable_names=["fALFF"])
    return (
        pl.DataFrame(postprocess["fALFF"], schema=["fALFF"])
        .with_columns(component=pl.int_range(pl.len(), dtype=pl.UInt16))
        .with_columns(
            sub=pl.lit(utils.get_sub(f)),
            ses=pl.lit(utils.get_ses(f)),
            task=pl.lit(utils.get_entity(f, "(?<=task-)[a-z]+")),
            run=int(utils.get_entity(f, r"(?<=run-)\d+")),
            model=model_from_path(f),
        )
    )


def timecourse_from_mat(mat: Path) -> Path:
    sub = utils.get_sub(mat)
    ses = utils.get_ses(mat)
    task = utils.get_entity(mat, r"(?<=task-)[a-z]+")
    run = utils.get_entity(mat, r"(?<=run-)\d+")
    modelorder_dir = mat.parent.parent.parent.parent
    return (
        modelorder_dir
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "func"
        / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_bold_timecourses.nii.gz"
    )


def postgift_flow(indir: Path, outdir: Path) -> None:
    fcns: list[pl.DataFrame] = []
    falffs: list[pl.DataFrame] = []
    for f in indir.rglob("*sub_001.mat"):
        logging.info(f"Extracting IDPs from {f}")
        fcns.append(extract_fcn_corrs(f))
        falffs.append(extract_falff(f))

    pl.concat(falffs).lazy().sink_parquet(
        pl.PartitionByKey(
            outdir / "amplitude", by=[pl.col.sub, pl.col.ses, pl.col.model]
        ),
        mkdir=True,
    )
    pl.concat(fcns).lazy().sink_parquet(
        pl.PartitionByKey(
            outdir / "connectivity", by=[pl.col.sub, pl.col.ses, pl.col.model]
        ),
        mkdir=True,
    )

    biomarkers = {
        "sub": [],
        "ses": [],
        "task": [],
        "run": [],
        "dmn_insula": [],
        "dmn_s1": [],
        "sln_s1m1": [],
    }
    for f in (
        indir
        / "derivatives"
        / "gift-neuromark_fmri_2.1_modelorder-multi"
        / "derivatives"
    ).rglob("*sub_001.mat"):
        logging.info(f"Adding biomarkers from {f}")
        biomarkers["sub"].append(np.uint16(utils.get_sub(f)))
        biomarkers["ses"].append(utils.get_ses(f))
        biomarkers["task"].append(utils.get_entity(f, r"(?<=task-)[a-z]+"))
        biomarkers["run"].append(np.uint8(utils.get_entity(f, r"(?<=run-)\d+")))

        timecourse = timecourse_from_mat(f)
        nii = reconstruct_nifti(timecourse)
        biomarkers["dmn_insula"].append(extract_dmn_insula(timecourse, nii=nii))
        biomarkers["dmn_s1"].append(
            extract_component_map_connectivity(
                timecourse, nii, DMN_COMPONENT, map=datasets.get_s1()
            )
        )
        gc.collect()
        utils.trim_memory()
        biomarkers["sln_s1m1"].append(
            extract_component_map_connectivity(
                timecourse, nii, SLN_COMPONENT, map=datasets.get_s1m1()
            )
        )
        gc.collect()
        utils.trim_memory()

    pl.DataFrame(biomarkers).lazy().sink_parquet(
        pl.PartitionByKey(outdir / "biomarkers", by=[pl.col.sub, pl.col.ses]),
        mkdir=True,
    )
