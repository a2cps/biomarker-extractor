import nipype
from ..interfaces.fslanat import FSLAnat

from ..nodes import io

# TODO: generate report to check registration


def get_volumes(
    src: str,
    labels: list[str] = [
        "Left-Amygdala",
        "Right-Amygdala",
        "Left-Hippocampus",
        "Right-Hippocampus",
    ],
) -> str:
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import nibabel as nib

    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide#Labels
    rois = {
        "Left-Thalamus-Proper": 10,
        "Left-Caudate": 11,
        "Left-Putamen": 12,
        "Left-Pallidum": 13,
        "Brain-Stem /4th Ventricle": 16,
        "Left-Hippocampus": 17,
        "Left-Amygdala": 18,
        "Left-Accumbens-area": 26,
        "Right-Thalamus-Proper": 49,
        "Right-Caudate": 50,
        "Right-Putamen": 51,
        "Right-Pallidum": 52,
        "Right-Hippocampus": 53,
        "Right-Amygdala": 54,
        "Right-Accumbens-area": 58,
    }
    srcpath = Path(src)
    srcpath_stem = srcpath.name.removesuffix(".anat")
    segmentation_file = srcpath / "T1_subcort_seg.nii.gz"
    nii = np.asanyarray(nib.load(segmentation_file).get_fdata(), dtype=np.uint8)
    vol_dict = {}
    for region in labels:
        vol_dict.update({region: np.sum(nii == rois.get(region), dtype=np.uint32)})
    out = pd.DataFrame.from_dict(
        vol_dict, orient="index", columns=["volume"]
    ).rename_axis("region")
    out["src"] = srcpath_stem
    out["method"] = "FIRST"
    f = Path(f"{srcpath_stem}.tsv").absolute()
    out.to_csv(f, sep="\t")
    return str(f)


class FIRSTWF(nipype.Workflow):
    def __init__(self, **inputs):
        super().__init__(name="first_wf", **inputs)
        inputnode = io.InputNode.from_fields(["in_file"])
        outputnode = io.OutputNode.from_fields(["anat", "volumes"])
        first = nipype.Node(interface=FSLAnat(), name="fsl_anat")
        volumes = nipype.Node(
            interface=nipype.Function(
                input_names=["src", "labels"],
                output_names=["volumes"],
                function=get_volumes,
            ),
            name="volumes",
        )
        self.connect(
            [
                (inputnode, first, [("in_file", "in_file")]),
                (first, volumes, [("anat", "src")]),
                (volumes, outputnode, [("volumes", "volumes")]),
                (first, outputnode, [("anat", "anat")]),
            ]
        )
