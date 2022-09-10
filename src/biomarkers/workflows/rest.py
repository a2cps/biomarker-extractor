from __future__ import annotations

import nipype
from nipype.interfaces import fsl

from ..interfaces import connectivity
from ..interfaces import fslutils
from ..nodes import io


class RestWF(nipype.Workflow):
    def __init__(self, **inputs) -> RestWF:
        super().__init__(name="rest", **inputs)
        inputnode = io.InputNode.from_fields(["in_file", "anat"])
        outputnode = io.OutputNode.from_fields(["correlation_matrix", "feat_dir"])

        bet = nipype.Node(fsl.BET(out_file="brain.nii.gz"), name="bet")
        feat = nipype.Node(fslutils.FEAT2(), name="feat")
        featimg = nipype.Node(fslutils.FEATImg(), name="featimg")

        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FNIRT/UserGuide#Transforming_a_functional_image_into_standard_space
        warp = nipype.Node(
            fsl.ApplyWarp(
                ref_file=fsl.base.Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
            ),
            name="applywarp",
        )
        confounds = nipype.Node(fslutils.Confounds(), name="confounds")
        fc = nipype.Node(connectivity.FEATCon(), name="fc")
        self.connect(
            [
                (
                    inputnode,
                    feat,
                    [
                        ("in_file", "in_file"),
                        ("anat", "anat_file"),
                    ],
                ),
                (inputnode, bet, [("anat", "in_file")]),
                (bet, feat, [("out_file", "brain_file")]),
                (feat, featimg, [("outputdir", "feat_dir")]),
                (
                    featimg,
                    warp,
                    [
                        ("filtered_func_data", "in_file"),
                        ("example_func2standard_warp", "field_file"),
                    ],
                ),
                (warp, confounds, [("out_file", "in_file")]),
                (feat, confounds, [("outputdir", "feat_dir")]),
                (confounds, fc, [("confounds", "confounds")]),
                (warp, fc, [("out_file", "in_file")]),
                (
                    fc,
                    outputnode,
                    [("correlation_matrix", "correlation_matrix")],
                ),
                (feat, outputnode, [("outputdir", "feat_dir")]),
            ]
        )
