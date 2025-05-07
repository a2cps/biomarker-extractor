from pathlib import Path

import bct
import nibabel as nb
import numpy as np
import polars as pl
import pydantic

from biomarkers import utils


class Module(pydantic.BaseModel):
    index: int
    name: str
    is_biomarker: bool


MODULE_INFO = (
    Module(index=1, name="mPFCventral-amyg", is_biomarker=False),
    Module(index=2, name="OFC-amyg-hipp", is_biomarker=False),
    Module(index=3, name="mPFCdorsal-amyg-NAc", is_biomarker=True),
)


def get_adjacency_matrix(coords: np.ndarray, src: Path) -> np.ndarray:
    tracts_img = nb.nifti1.Nifti1Image.load(src).get_fdata()

    # Number of voxels
    nvox = coords.shape[0]

    tracts_mat = np.zeros((nvox, nvox))
    for i, (x, y, z, _) in enumerate(coords):
        tracts_mat[i, :] = tracts_img[x, y, z, :].squeeze()

    return tracts_mat


def symmetrize(adjacency: np.ndarray) -> np.ndarray:
    # Symmetrize and zero-diagonal
    adjacency_sym = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency_sym, 0)
    return adjacency_sym


def get_summary(mat: np.ndarray) -> pl.DataFrame:
    Eglob = bct.efficiency_bin(mat)
    Ccoef = bct.clustering_coef_bu(mat).mean()
    Betw = bct.betweenness_bin(mat).mean()
    Dist = 1.0 / Eglob
    Deg = bct.degrees_und(mat)
    Deg_mean = np.mean(Deg)
    density, _, _ = bct.density_und(mat)
    nvox = mat.shape[0]
    WM_conn = np.sum(Deg) / (nvox * (nvox - 1) / 2)
    return pl.DataFrame(
        {
            "Eglob": Eglob,
            "Ccoef": Ccoef,
            "Betw": Betw,
            "Dist": Dist,
            "Deg_mean": Deg_mean,
            "Density": density,
            "WM_connections": WM_conn,
        }
    )


def write_adjacency_as_df(adjacency: np.ndarray, dst: Path):
    utils.mkdir_recursive(dst.parent)
    (
        pl.DataFrame(adjacency)
        .with_row_index("source")
        .unpivot(index="source", variable_name="target", value_name="connectivity")
        .with_columns(pl.col("target").str.strip_prefix("column_").cast(pl.UInt32()))
        .write_parquet(dst)
    )


def dwi_biomarker1_flow(
    outdir: Path,
    participant_label: str,
    session_label: str,
    target_density: float = 0.1,
) -> None:
    # Paths
    roi_path = outdir / "move_masks" / participant_label / session_label
    tract_path = (
        outdir
        / "probtrackx"
        / participant_label
        / session_label
        / "DWIbiomarker1_modules_all_voxseeds"
    )

    # ------------------------------
    # Step 1: Load mask and get voxel coordinates
    # ------------------------------
    mask_img = nb.nifti1.Nifti1Image.load(
        roi_path
        / f"{participant_label}_{session_label}_desc-mask_modules_all_index_space-dwi-fslstd.nii.gz"
    )
    dmask = np.asarray(mask_img.get_fdata(), dtype=np.int64)

    # Collect voxel indices and ROI labels
    coords = []
    for roi_id in range(len(MODULE_INFO)):
        _coords = np.argwhere(dmask == roi_id + 1)
        coords.append(np.column_stack([_coords, np.repeat(roi_id, _coords.shape[0])]))

    # Combine to coor_roi: x, y, z, roi_id
    coor_roi = np.vstack(coords)[1:]

    # ------------------------------
    # Step 2: Build distance matrix
    # ------------------------------

    lengths = get_adjacency_matrix(
        coor_roi,
        tract_path
        / f"{participant_label}_{session_label}_DWIbiomarker1_fdtlengths_all.nii.gz",
    )

    # ------------------------------
    # Step 3: Build tractography (streamline count) matrix
    # ------------------------------
    paths = get_adjacency_matrix(
        coor_roi,
        tract_path
        / f"{participant_label}_{session_label}_DWIbiomarker1_fdtpaths_all.nii.gz",
    )

    # Distance correction: multiply by distances
    final_bin = symmetrize(paths) * symmetrize(lengths)

    write_adjacency_as_df(
        paths,
        outdir
        / "connectivity"
        / f"sub={participant_label.removeprefix('sub-')}"
        / f"ses={session_label.removeprefix('ses-')}"
        / "part-0.parquet",
    )

    # Normalize weights
    bct.weight_conversion(final_bin, "normalize", copy=False)

    # ------------------------------
    # Step 4: Threshold to target density
    # ------------------------------

    bct.weight_conversion(
        bct.threshold_proportional(final_bin, target_density, copy=False),
        "binarize",
        copy=False,
    )

    # ------------------------------
    # Step 5a: Module-level summary
    # ------------------------------
    rows: list[pl.DataFrame] = []
    for module in MODULE_INFO:
        inds = np.asarray(coor_roi[:, 3] == module.index).nonzero()[0]
        mat = final_bin[np.ix_(inds, inds)]
        rows.append(
            get_summary(mat).with_columns(
                ModuleNumber=module.index,
                ModuleName=pl.lit(module.name),
                IsBiomarker=module.is_biomarker,
            )
        )

    # ------------------------------
    # Step 5b: Whole-network summary
    # ------------------------------
    rows.append(
        get_summary(final_bin).with_columns(
            ModuleNumber=0, ModuleName=pl.lit("WholeNetwork"), IsBiomarker=False
        )
    )
    out: pl.DataFrame = pl.concat(rows).with_columns(
        sub=pl.lit(participant_label).str.strip_prefix("sub-"),
        ses=pl.lit(session_label).str.strip_prefix("ses-"),
    )
    out.write_csv(tract_path / "network_summaries.tsv", separator="\t")
