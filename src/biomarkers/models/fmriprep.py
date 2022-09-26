from pathlib import Path

from pydantic.dataclasses import dataclass

from . import bids


@dataclass(frozen=True)
class Anat:
    desc_aparcaseg_dseg: bids.File
    desc_aseg_dseg: bids.File
    desc_brain_mask: bids.Pair
    desc_preproc_T1w: bids.Pair
    dseg: bids.File
    from_fsnative_to_T1w_mode_image_xfm: bids.File
    from_MNI152NLin6Asym_to_T1w_mode_image_xfm: bids.File
    from_MNI152NLin2009cAsym_to_T1w_mode_image_xfm: bids.File
    from_T1w_to_fsnative_mode_image_xfm: bids.File
    from_T1w_to_MNI152NLin6Asym_mode_image_xfm: bids.File
    from_T1w_to_MNI152NLin2009cAsym_mode_image_xfm: bids.File
    label_CSF_probseg: bids.File
    label_GM_probseg: bids.File
    label_WM_probseg: bids.File
    space_MNI152NLin2009cAsym_desc_brain_mask: bids.Pair
    space_MNI152NLin2009cAsym_desc_preproc_T1w: bids.Pair
    space_MNI152NLin2009cAsym_dseg: bids.File
    space_MNI152NLin2009cAsym_label_CSF_probseg: bids.File
    space_MNI152NLin2009cAsym_label_GM_probseg: bids.File
    space_MNI152NLin2009cAsym_label_WM_probseg: bids.File

    @classmethod
    def from_root(cls, root: Path, sub: str, ses: str) -> "Anat":
        return cls(
            desc_aparcaseg_dseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="dseg",
                extension="nii.gz",
                modality="anat",
            ),
            desc_aseg_dseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aseg",
                suffix="dseg",
                extension="nii.gz",
                modality="anat",
            ),
            desc_brain_mask=bids.Pair.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="brain",
                suffix="mask",
                extension="nii.gz",
                modality="anat",
            ),
            desc_preproc_T1w=bids.Pair.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="preproc",
                suffix="T1w",
                extension="nii.gz",
                modality="anat",
            ),
            dseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                suffix="dseg",
                extension="nii.gz",
                modality="anat",
            ),
            from_fsnative_to_T1w_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("fsnative", "T1w"),
                mode="image",
                extension="txt",
                modality="anat",
            ),
            from_MNI152NLin6Asym_to_T1w_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("MNI152NLin6Asym", "T1w"),
                mode="image",
                extension="h5",
                modality="anat",
            ),
            from_MNI152NLin2009cAsym_to_T1w_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("MNI152NLin2009cAsym", "T1w"),
                mode="image",
                extension="h5",
                modality="anat",
            ),
            from_T1w_to_fsnative_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("T1w", "fsnative"),
                mode="image",
                extension="txt",
                modality="anat",
            ),
            from_T1w_to_MNI152NLin6Asym_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("T1w", "MNI152NLin6Asym"),
                mode="image",
                extension="h5",
                modality="anat",
            ),
            from_T1w_to_MNI152NLin2009cAsym_mode_image_xfm=bids.Transform.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                fromto=("T1w", "MNI152NLin2009cAsym"),
                mode="image",
                extension="h5",
                modality="anat",
            ),
            label_CSF_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="CSF",
            ),
            label_GM_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="GM",
            ),
            label_WM_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="WM",
            ),
            space_MNI152NLin2009cAsym_desc_brain_mask=bids.Pair.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="brain",
                suffix="mask",
                extension="nii.gz",
                modality="anat",
                space="MNI152NLin2009cAsym",
            ),
            space_MNI152NLin2009cAsym_desc_preproc_T1w=bids.Pair.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="preproc",
                suffix="T1w",
                extension="nii.gz",
                modality="anat",
                space="MNI152NLin2009cAsym",
            ),
            space_MNI152NLin2009cAsym_dseg=bids.Pair.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                suffix="dseg",
                extension="nii.gz",
                modality="anat",
                space="MNI152NLin2009cAsym",
            ),
            space_MNI152NLin2009cAsym_label_CSF_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="CSF",
                space="MNI152NLin2009cAsym",
            ),
            space_MNI152NLin2009cAsym_label_GM_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="GM",
                space="MNI152NLin2009cAsym",
            ),
            space_MNI152NLin2009cAsym_label_WM_probseg=bids.File.from_entities(
                root=root,
                sub=sub,
                ses=ses,
                desc="aparcaseg",
                suffix="probseg",
                extension="nii.gz",
                modality="anat",
                label="WM",
                space="MNI152NLin2009cAsym",
            ),
        )


@dataclass(frozen=True)
class Bold:
    pass


@dataclass(frozen=True)
class Derivative:
    anat: Anat
    bold: Bold | None

    @classmethod
    def from_root(cls, root: Path, sub: str, ses: str | None = None) -> "Derivative":

        return cls(
            anat=Anat.from_root(root=root, sub=sub, ses=ses),
        )
