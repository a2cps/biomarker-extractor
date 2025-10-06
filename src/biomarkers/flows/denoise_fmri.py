import functools
import logging
import shutil
import tempfile
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import polars as pl
import pydantic
from nilearn import image, masking
from nitransforms import io
from niwrap_ants import ants
from niwrap_fsl import fsl

from biomarkers import datasets, utils
from biomarkers.models import bids, fmriprep

N_TRS_EXPECTED = 450
N_NON_STEADY_STATE = 15
MEDIAN_INTENSITY_TARGET = 10000
HP_SIGMA = 100
SPACE = "MNI152NLin6Asym"
FIX_THRESHOLD = 20


class MELODICPrepper(pydantic.BaseModel):
    dst: Path
    sub: str
    ses: str
    layout: bids.Layout
    task: str
    run: str
    space: fmriprep.SPACE | None = None
    res: str | None = None

    @property
    def filter(self) -> dict[str, str]:
        return {"sub": self.sub, "ses": self.ses, "task": self.task, "run": self.run}

    @property
    def filter_with_space(self) -> dict[str, str | None]:
        return {**self.filter, "space": self.space, "resolution": self.res}

    @property
    def anat(self) -> Path:
        return self.layout.get(
            filters={
                "sub": self.sub,
                "ses": self.ses,
                "desc": "preproc",
                "suffix": "T1w",
                "space": self.space,
                "resolution": self.res,
                "extension": ".nii.gz",
            }
        )

    @property
    def dseg(self) -> Path:
        return self.anat.with_name(self.anat.name.replace("desc-preproc_T1w", "dseg"))

    @property
    def preproc(self) -> Path:
        return self.layout.get_preproc(filters=self.filter_with_space)

    @property
    def boldref(self) -> Path:
        return self.layout.get_boldref(filters=self.filter_with_space)

    @property
    def brain(self) -> Path:
        return self.preproc.with_name(
            self.preproc.name.replace("desc-preproc_bold", "desc-brain_mask")
        )

    @property
    def confounds(self) -> Path:
        return self.layout.get_confounds(filters=self.filter)

    @property
    def outdir(self) -> Path:
        return self.dst / "melodic" / utils.img_stem(self.preproc)

    @property
    def mask(self) -> Path:
        return self.outdir / "mask.nii.gz"

    @property
    def mcdir(self) -> Path:
        return self.outdir / "mc"

    @property
    def regdir(self) -> Path:
        return self.outdir / "reg"

    @property
    def filtered_func(self) -> Path:
        return self.outdir / "filtered_func_data.nii.gz"

    @property
    def smoothed_func(self) -> Path:
        return self.outdir / "filtered_func_data_smooth.nii.gz"

    @property
    def highres(self) -> Path:
        return self.regdir / "highres.nii.gz"

    @property
    def example_func(self) -> Path:
        return self.regdir / "example_func.nii.gz"

    @property
    def standard(self) -> Path:
        return self.regdir / "standard.nii.gz"

    @property
    def example_func2highres_mat(self) -> Path:
        return self.regdir / "example_func2highres.mat"

    @property
    def highres2example_func_mat(self) -> Path:
        return self.regdir / "highres2example_func.mat"

    @property
    def highres2standard_mat(self) -> Path:
        return self.regdir / "highres2standard.mat"

    @property
    def example_func2standard_mat(self) -> Path:
        return self.regdir / "example_func2standard.mat"

    @property
    def highres2standard_warp(self) -> Path:
        return self.regdir / "highres2standard_warp.nii.gz"

    @property
    def example_func2standard_warp(self) -> Path:
        return self.regdir / "example_func2standard_warp.nii.gz"

    @property
    def standard2example_func_warp(self) -> Path:
        return self.regdir / "standard2example_func_warp.nii.gz"

    @property
    def wmparc(self) -> Path:
        return self.regdir / "wmparc.nii.gz"

    @property
    def highres_pveseg(self) -> Path:
        return self.regdir / "highres_pveseg.nii.gz"

    # prefiltered_func_data_mcf
    @property
    def prefiltered_func_data_mcf(self) -> Path:
        return self.mcdir / "prefiltered_func_data_mcf.par"

    @property
    def boldref_to_t1w(self) -> Path:
        return self.layout.get(
            {
                "sub": self.sub,
                "ses": self.ses,
                "task": self.task,
                "run": self.run,
                "suffix": "xfm",
                "desc": "coreg",
                "extension": "txt",
            }
        )

    @property
    def t1w_to_standard(self) -> Path:
        return (
            self.anat.parent
            / f"sub-{self.sub}_ses-{self.ses}_from-T1w_to-{SPACE}_mode-image_xfm.h5"
        )

    @property
    def fsnative_to_t1(self) -> Path:
        return (
            self.anat.parent
            / f"sub-{self.sub}_ses-{self.ses}_from-fsnative_to-T1w_mode-image_xfm.txt"
        )

    @functools.cached_property
    def probseg(self) -> bids.ProbSeg:
        filters = {"sub": self.sub, "ses": self.ses}
        if self.space is not None:
            filters = {
                **filters,
                "space": self.space,
                "resolution": self.res,
            }

        return bids.ProbSeg.from_layout(self.layout, filters)


async def do_melodic(func: Path, mask: Path, log: Path) -> None:
    async with utils.subprocess_manager(
        log=log,
        args=[
            "melodic",
            "-i",
            str(func),
            "-m",
            str(mask),
            "--Oall",
            f"--tr={utils.get_tr(func)}",
            "--mmthresh=0.5",
            "--report",
        ],
    ) as proc:
        await proc.wait()
        # if proc.returncode is None or proc.returncode > 0:
        #     msg = f"melodic failed with {proc.returncode=}"
        #     raise RuntimeError(msg)


async def denoise_flow(subdir: Path, out: Path) -> None:
    for sdir in subdir.glob("sub*"):
        if not sdir.is_dir():
            continue
        layout = bids.Layout.from_path(subdir)
        sub = utils.get_sub(sdir)
        for ses in layout.get_sessions(sub=sub):
            freesurfer_dir = (
                subdir / "sourcedata" / "freesurfer" / f"sub-{sub}_ses-{ses}"
            )
            wmparc_src = freesurfer_dir / "mri" / "wmparc.mgz"
            for task in layout.get_tasks(sub=sub, ses=ses):
                for run in layout.get_runs(sub=sub, ses=ses, task=task):
                    prepper = MELODICPrepper(
                        dst=out, sub=sub, ses=ses, layout=layout, task=task, run=run
                    )
                    ## prep for melodic
                    preproc = nb.nifti1.Nifti1Image.load(prepper.preproc)
                    if not (preproc.shape[-1] == N_TRS_EXPECTED):
                        logging.warning(f"{prepper.preproc} has too few trs. skipping")
                        continue
                    logging.info(f"prepping melodic for {prepper.preproc}")

                    prefiltered_func_data = masking.apply_mask(
                        preproc,
                        prepper.brain,
                        dtype=np.float64,  # type: ignore
                    )[N_NON_STEADY_STATE:, :]

                    median_intensity = np.median(prefiltered_func_data)
                    prefiltered_func_data *= MEDIAN_INTENSITY_TARGET / median_intensity

                    if not prepper.outdir.exists():
                        prepper.outdir.mkdir(parents=True)

                    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as prefiltered:
                        # this will be added back in after temporal filtering
                        # we're doing these extra steps because the bptfm isn't quite the same
                        # as the current approach of adding in the mean
                        prefiltered_func = image.new_img_like(  # type: ignore
                            preproc,
                            masking.unmask(
                                prefiltered_func_data, prepper.brain
                            ).get_fdata(),  # type: ignore
                            affine=preproc.affine,
                            copy_header=True,
                        )
                        prefiltered_func.to_filename(prefiltered.name)  # type: ignore
                        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmean:
                            fsl.fslmaths(
                                input_files=[prefiltered.name],
                                operations=[fsl.fslmaths_operation_params(tmean=True)],
                                output=tmean.name,
                                datatype_internal="double",
                                output_datatype="double",
                            )
                            fsl.fslmaths(
                                input_files=[prefiltered.name],
                                operations=[
                                    fsl.fslmaths_operation_params(
                                        bptf=[
                                            HP_SIGMA / (2 * utils.get_tr(preproc)),
                                            -1,
                                        ]
                                    ),
                                    fsl.fslmaths_operation_params(add=tmean.name),
                                ],
                                output=str(prepper.filtered_func),
                                datatype_internal="double",
                                output_datatype="double",
                            )

                    shutil.copy2(prepper.brain, prepper.mask)

                    ## run melodic
                    # for visualization, the spatial maps will be smoothed
                    # don't smooth beforehand, otherwise the individual components
                    # might not match between smooth/unsmoothed
                    await do_melodic(
                        func=prepper.filtered_func,
                        mask=prepper.mask,
                        log=Path("/dev/null"),  # melodic generates its own mask
                    )

                    if not prepper.regdir.exists():
                        prepper.regdir.mkdir(parents=True)

                    if not prepper.mcdir.exists():
                        prepper.mcdir.mkdir(parents=True)

                    ## prep fix
                    shutil.copy2(prepper.boldref, prepper.example_func)
                    shutil.copy2(prepper.anat, prepper.highres)

                    ants.ants_apply_transforms(
                        reference_image=prepper.highres,
                        output=ants.ants_apply_transforms_warped_output_params(
                            str(prepper.wmparc)
                        ),
                        dimensionality=3,
                        input_image=wmparc_src,
                        interpolation=ants.ants_apply_transforms_generic_label_params(
                            "GenericLabel"
                        ),
                        transform=[
                            ants.ants_apply_transforms_transform_file_name_params(
                                prepper.fsnative_to_t1
                            )
                        ],
                    )

                    template = datasets.get_standard_file()
                    if not prepper.standard.exists():
                        shutil.copy2(template, prepper.standard)

                    nt.linear.load(
                        prepper.boldref_to_t1w,
                        reference=prepper.highres,
                        moving=prepper.example_func,
                    ).to_filename(prepper.example_func2highres_mat, fmt="fsl")

                    xfm = io.itk.ITKCompositeH5.from_filename(prepper.t1w_to_standard)
                    nt.linear.Affine(
                        xfm[0].to_ras(
                            reference=prepper.standard, moving=prepper.highres
                        ),
                        reference=prepper.standard,
                    ).to_filename(
                        prepper.highres2standard_mat,
                        fmt="fsl",
                        moving=prepper.highres,
                    )

                    io.fsl.FSLDisplacementsField.from_image(xfm[1]).to_filename(
                        prepper.highres2standard_warp
                    )

                    fsl.convert_xfm(
                        in_file=prepper.example_func2highres_mat,
                        out_file=str(prepper.highres2example_func_mat),
                        invert_xfm=True,
                    )

                    fsl.convert_xfm(
                        in_file=prepper.example_func2highres_mat,
                        out_file=str(prepper.example_func2standard_mat),
                        concat_xfm=prepper.highres2standard_mat,
                    )

                    async with utils.subprocess_manager(
                        log=Path("/dev/null"),
                        args=[
                            "convertwarp",
                            f"--ref={prepper.standard}",
                            f"--premat={prepper.example_func2highres_mat}",
                            f"--warp1={prepper.highres2standard_warp}",
                            f"--out={prepper.example_func2standard_warp}",
                        ],
                    ) as proc:
                        await proc.wait()

                    fsl.invwarp(
                        ref_img=prepper.example_func,
                        warp=prepper.example_func2standard_warp,
                        out_img=str(prepper.standard2example_func_warp),
                    )

                    shutil.copy2(prepper.dseg, prepper.highres_pveseg)

                    confounds = (
                        pl.scan_csv(
                            prepper.confounds, separator="\t", null_values="n/a"
                        )
                        .select(
                            "rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"
                        )
                        .tail(N_TRS_EXPECTED - N_NON_STEADY_STATE)
                        .collect()
                    )
                    np.savetxt(prepper.prefiltered_func_data_mcf, confounds.to_numpy())

                    ## run fix
                    # doing it here rather than with pyfix because otherwise their logging file is weird
                    async with utils.subprocess_manager(
                        log=out / "pyfix.log",
                        args=[
                            "fix",
                            f"{prepper.outdir}",
                            "UKBiobank",
                            str(FIX_THRESHOLD),
                            "-m",
                            "-h",
                            str(HP_SIGMA),
                            f"--logfile={prepper.outdir / 'pyfix.log'}",
                        ],
                    ) as proc:
                        await proc.wait()
                        if proc.returncode is None or proc.returncode > 0:
                            # remove folder so that archiving detects that there was a failure
                            # and sends logs to failure_dst_dir
                            msg = f"fix failed with {proc.returncode=}"
                            raise RuntimeError(msg)

                    # if (styx_tmp := Path("styx_tmp")).exists():
                    #     shutil.rmtree(styx_tmp)
