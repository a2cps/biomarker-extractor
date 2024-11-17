import re
import typing
from pathlib import Path
from typing import Iterable, Literal

import nibabel as nb
import numpy as np
import pandas as pd
import pydantic
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class FIRSTROI:
    name: str
    root: pydantic.DirectoryPath
    bvars: pydantic.FilePath
    vtk: pydantic.FilePath

    @classmethod
    def from_nameroot(cls, name: str, root: pydantic.DirectoryPath) -> typing.Self:
        return cls(
            name=name,
            root=root,
            bvars=root / f"T1_first-{name}_first.bvars",
            vtk=root / f"T1_first-{name}_first.vtk",
        )


@dataclass(frozen=True)
class FIRSTLabel:
    label: Literal[
        "Left-Thalamus-Proper",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "Brain-Stem /4th Ventricle",
        "Left-Hippocampus",
        "Left-Amygdala",
        "Left-Accumbens-area",
        "Right-Thalamus-Proper",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
    ]

    def __str__(self):
        return self.label


@dataclass(frozen=True)
class FIRSTResults:
    root: pydantic.DirectoryPath
    BrStem: FIRSTROI
    L_Accu: FIRSTROI
    R_Accu: FIRSTROI
    L_Amyg: FIRSTROI
    R_Amyg: FIRSTROI
    L_Caud: FIRSTROI
    R_Caud: FIRSTROI
    L_Hipp: FIRSTROI
    R_Hipp: FIRSTROI
    L_Pall: FIRSTROI
    R_Pall: FIRSTROI
    L_Puta: FIRSTROI
    R_Puta: FIRSTROI
    L_Thal: FIRSTROI
    R_Thal: FIRSTROI
    T1_first_all_fast_firstseg: pydantic.FilePath
    T1_first_all_fast_origsegs: pydantic.FilePath
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide#Labels
    # ValueError: mutable default <class 'dict'> for field rois is not allowed: use default_factory
    rois: dict[FIRSTLabel, int] = pydantic.Field(
        default_factory=lambda: {
            FIRSTLabel(label="Left-Thalamus-Proper"): 10,
            FIRSTLabel(label="Left-Caudate"): 11,
            FIRSTLabel(label="Left-Putamen"): 12,
            FIRSTLabel(label="Left-Pallidum"): 13,
            FIRSTLabel(label="Brain-Stem /4th Ventricle"): 16,
            FIRSTLabel(label="Left-Hippocampus"): 17,
            FIRSTLabel(label="Left-Amygdala"): 18,
            FIRSTLabel(label="Left-Accumbens-area"): 26,
            FIRSTLabel(label="Right-Thalamus-Proper"): 49,
            FIRSTLabel(label="Right-Caudate"): 50,
            FIRSTLabel(label="Right-Putamen"): 51,
            FIRSTLabel(label="Right-Pallidum"): 52,
            FIRSTLabel(label="Right-Hippocampus"): 53,
            FIRSTLabel(label="Right-Amygdala"): 54,
            FIRSTLabel(label="Right-Accumbens-area"): 58,
        }
    )

    @classmethod
    def from_root(cls, root: pydantic.DirectoryPath) -> typing.Self:
        return cls(
            root=root,
            BrStem=FIRSTROI.from_nameroot("BrStem", root=root),
            L_Accu=FIRSTROI.from_nameroot("L_Accu", root=root),
            R_Accu=FIRSTROI.from_nameroot("R_Accu", root=root),
            L_Amyg=FIRSTROI.from_nameroot("L_Amyg", root=root),
            R_Amyg=FIRSTROI.from_nameroot("R_Amyg", root=root),
            L_Caud=FIRSTROI.from_nameroot("L_Caud", root=root),
            R_Caud=FIRSTROI.from_nameroot("R_Caud", root=root),
            L_Hipp=FIRSTROI.from_nameroot("L_Hipp", root=root),
            R_Hipp=FIRSTROI.from_nameroot("R_Hipp", root=root),
            L_Pall=FIRSTROI.from_nameroot("L_Pall", root=root),
            R_Pall=FIRSTROI.from_nameroot("R_Pall", root=root),
            L_Puta=FIRSTROI.from_nameroot("L_Puta", root=root),
            R_Puta=FIRSTROI.from_nameroot("R_Puta", root=root),
            L_Thal=FIRSTROI.from_nameroot("L_Thal", root=root),
            R_Thal=FIRSTROI.from_nameroot("R_Thal", root=root),
            T1_first_all_fast_firstseg=Path(root / "T1_first_all_fast_firstseg.nii.gz"),
            T1_first_all_fast_origsegs=Path(root / "T1_first_all_fast_origsegs.nii.gz"),
        )


@dataclass(frozen=True)
class FSLAnatResult:
    root: pydantic.DirectoryPath
    first_results: FIRSTResults
    lesionmask: pydantic.FilePath
    lesionmaskinv: pydantic.FilePath
    MNI152_T1_2mm_brain_mask_dil1: pydantic.FilePath
    MNI_to_T1_nonlin_field: pydantic.FilePath
    T1: pydantic.FilePath
    T1_biascorr: pydantic.FilePath
    T1_biascorr_bet_skull: pydantic.FilePath
    T1_biascorr_brain: pydantic.FilePath
    T1_biascorr_brain_mask: pydantic.FilePath
    T1_biascorr_to_std_sub: pydantic.FilePath
    T1_fast_bias: pydantic.FilePath
    T1_fast_mixeltype: pydantic.FilePath
    T1_fast_pve_0: pydantic.FilePath
    T1_fast_pve_1: pydantic.FilePath
    T1_fast_pve_2: pydantic.FilePath
    T1_fast_pveseg: pydantic.FilePath
    T1_fast_restore: pydantic.FilePath
    T1_fast_seg: pydantic.FilePath
    T1_fullfov: pydantic.FilePath
    T1_nonroi2roi: pydantic.FilePath
    T1_orig: pydantic.FilePath
    T1_orig2roi: pydantic.FilePath
    T1_orig2std: pydantic.FilePath
    T1_roi2nonroi: pydantic.FilePath
    T1_roi2orig: pydantic.FilePath
    T1_std2orig: pydantic.FilePath
    T1_subcort_seg: pydantic.FilePath
    T1_to_MNI_lin_: pydantic.FilePath
    T1_to_MNI_lin: pydantic.FilePath
    T1_to_MNI_nonlin: pydantic.FilePath
    T1_to_MNI_nonlin_: pydantic.FilePath
    T1_to_MNI_nonlin_coeff: pydantic.FilePath
    T1_to_MNI_nonlin_field: pydantic.FilePath
    T1_to_MNI_nonlin_jac: pydantic.FilePath
    T1_vols: pydantic.FilePath
    T12std_skullcon: pydantic.FilePath

    @classmethod
    def from_root(cls, root: pydantic.DirectoryPath) -> typing.Self:
        return cls(
            root=root,
            first_results=FIRSTResults.from_root(root / "first_results"),
            lesionmask=root / "lesionmask.nii.gz",
            lesionmaskinv=root / "lesionmaskinv.nii.gz",
            MNI152_T1_2mm_brain_mask_dil1=root / "MNI152_T1_2mm_brain_mask_dil1.nii.gz",
            MNI_to_T1_nonlin_field=root / "MNI_to_T1_nonlin_field.nii.gz",
            T1=root / "T1.nii.gz",
            T1_biascorr=root / "T1_biascorr.nii.gz",
            T1_biascorr_bet_skull=Path(root / "T1_biascorr_bet_skull.nii.gz"),
            T1_biascorr_brain=Path(root / "T1_biascorr_brain.nii.gz"),
            T1_biascorr_brain_mask=Path(root / "T1_biascorr_brain_mask.nii.gz"),
            T1_biascorr_to_std_sub=Path(root / "T1_biascorr_to_std_sub.mat"),
            T1_fast_bias=Path(root / "T1_fast_bias.nii.gz"),
            T1_fast_mixeltype=Path(root / "T1_fast_mixeltype.nii.gz"),
            T1_fast_pve_0=Path(root / "T1_fast_pve_0.nii.gz"),
            T1_fast_pve_1=Path(root / "T1_fast_pve_1.nii.gz"),
            T1_fast_pve_2=Path(root / "T1_fast_pve_2.nii.gz"),
            T1_fast_pveseg=Path(root / "T1_fast_pveseg.nii.gz"),
            T1_fast_restore=Path(root / "T1_fast_restore.nii.gz"),
            T1_fast_seg=Path(root / "T1_fast_seg.nii.gz"),
            T1_fullfov=Path(root / "T1_fullfov.nii.gz"),
            T1_nonroi2roi=Path(root / "T1_nonroi2roi.mat"),
            T1_orig=Path(root / "T1_orig.nii.gz"),
            T1_orig2roi=Path(root / "T1_orig2roi.mat"),
            T1_orig2std=Path(root / "T1_orig2std.mat"),
            T1_roi2nonroi=Path(root / "T1_roi2nonroi.mat"),
            T1_roi2orig=Path(root / "T1_roi2orig.mat"),
            T1_std2orig=Path(root / "T1_std2orig.mat"),
            T1_subcort_seg=Path(root / "T1_subcort_seg.nii.gz"),
            T1_to_MNI_lin_=Path(root / "T1_to_MNI_lin.mat"),
            T1_to_MNI_lin=Path(root / "T1_to_MNI_lin.nii.gz"),
            T1_to_MNI_nonlin=Path(root / "T1_to_MNI_nonlin.nii.gz"),
            T1_to_MNI_nonlin_=Path(root / "T1_to_MNI_nonlin.txt"),
            T1_to_MNI_nonlin_coeff=Path(root / "T1_to_MNI_nonlin_coeff.nii.gz"),
            T1_to_MNI_nonlin_field=Path(root / "T1_to_MNI_nonlin_field.nii.gz"),
            T1_to_MNI_nonlin_jac=Path(root / "T1_to_MNI_nonlin_jac.nii.gz"),
            T1_vols=Path(root / "T1_vols.txt"),
            T12std_skullcon=Path(root / "T12std_skullcon.mat"),
        )

    def get_volumes(
        self,
        labels: Iterable[FIRSTLabel] = (
            FIRSTLabel(label="Left-Thalamus-Proper"),
            FIRSTLabel(label="Left-Caudate"),
            FIRSTLabel(label="Left-Putamen"),
            FIRSTLabel(label="Left-Pallidum"),
            FIRSTLabel(label="Brain-Stem /4th Ventricle"),
            FIRSTLabel(label="Left-Hippocampus"),
            FIRSTLabel(label="Left-Amygdala"),
            FIRSTLabel(label="Left-Accumbens-area"),
            FIRSTLabel(label="Right-Thalamus-Proper"),
            FIRSTLabel(label="Right-Caudate"),
            FIRSTLabel(label="Right-Putamen"),
            FIRSTLabel(label="Right-Pallidum"),
            FIRSTLabel(label="Right-Hippocampus"),
            FIRSTLabel(label="Right-Amygdala"),
            FIRSTLabel(label="Right-Accumbens-area"),
        ),
    ) -> pd.DataFrame:
        """Write the volumes to a csv."""
        nii = np.asanyarray(
            nb.nifti1.load(self.first_results.T1_first_all_fast_firstseg).get_fdata(),
            dtype=np.uint8,
        )
        vol_dict = {}
        for region in labels:
            vol_dict.update(
                {
                    str(region): np.sum(
                        nii == self.first_results.rois.get(region),
                        dtype=np.uint32,
                    )
                }
            )
        volumes = (
            pd.DataFrame.from_dict(vol_dict, orient="index", columns=["volume"])
            .rename_axis("region")
            .reset_index()
        )
        volumes["src"] = self.root.name
        return (
            volumes.pivot(columns="region", index="src", values="volume")
            .reset_index()
            .rename_axis(None, axis=1)
        )

    def get_t1volumes(self) -> pd.DataFrame:
        lines = self.T1_vols.read_text().splitlines()
        return pd.DataFrame(
            {
                "t1_to_mni_scaling": [float(re.findall(r"\d+\.\d+", lines[0])[0])],
                "t1_space_orig_volume": [
                    int(
                        re.findall(r" \d+", lines[1])[0]
                    )  # NOTE: space in pattern required
                ],
                "t1_space_mni_volume": [float(re.findall(r"\d+\.\d+", lines[2])[0])],
            }
        )
