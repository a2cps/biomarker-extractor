import logging
import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import fslanat as fslanat_models


def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return (out / basename).with_suffix(".anat").absolute()


def _reorient2standard(t1: Path) -> None:
    """_summary_

    Args:
        t1 (Path): _description_

    Description:
        run $FSLDIR/bin/fslmaths ${T1} ${T1}_orig
        run $FSLDIR/bin/fslreorient2std ${T1} > ${T1}_orig2std.mat
        run $FSLDIR/bin/convert_xfm -omat ${T1}_std2orig.mat -inverse ${T1}_orig2std.mat
        run $FSLDIR/bin/fslreorient2std ${T1} ${T1}
    """
    t1_std2orig = t1.with_name("T1_std2orig.mat")
    t1_orig2std = t1.with_name("T1_orig2std.mat")
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/fslmaths",
            f"{t1}",
            f"{t1.with_name('T1_orig')}",
        ]
    )
    utils.run_and_log_stdout(
        [f"{os.getenv('FSLDIR')}/bin/fslreorient2std", f"{t1}"],
        log=t1_orig2std,
    )
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{t1_std2orig}",
            "-inverse",
            f"{t1_orig2std}",
        ]
    )
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/fslreorient2std",
            f"{t1}",
            f"{t1}",
        ]
    )


def _precrop(anatdir: Path):
    """
    im=failing
    outputname=out
    i90=$( fslstats $im -P 90 )
    fslmaths $im -mul 0 -randn -mul $( echo "$i90 * 0.005" | bc -l ) -mas $im -add $im ${im}_noisy
    roivals=$( robustfov -i ${im}_noisy | grep 0 )
    fslroi $im $outputname $roivals
    """
    t1 = anatdir / "T1.nii.gz"

    # REORIENTATION 2 STANDARD
    _reorient2standard(t1)

    # AUTOMATIC CROPPING
    fullfov = t1.with_name("T1_fullfov.nii.gz")
    shutil.move(t1, fullfov)
    nii = nb.loadsave.load(fullfov)
    nonzero_i = np.flatnonzero(nii.get_fdata())
    sigma = np.quantile(nii.get_fdata().flat[nonzero_i], 0.9) * 0.005
    noisy: np.ndarray = nii.get_fdata().copy()
    noisy.flat[nonzero_i] += np.random.normal(0, sigma, size=nonzero_i.shape)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmpfile:
        nb.ni1.Nifti1Image(
            dataobj=noisy, affine=nii.affine, header=nii.header
        ).to_filename(tmpfile.name)

        log = subprocess.run(
            [  # noqa: S603
                f"{os.getenv('FSLDIR')}/bin/robustfov",
                "-m",
                f"{anatdir / 'T1_roi2nonroi.mat'}",
                "-i",
                f"{tmpfile.name}",
            ],
            capture_output=True,
            text=True,
        )
    roi = log.stdout.splitlines()[1]
    (anatdir / "T1_roi.log").write_text(roi)

    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/fslroi",
            f"{fullfov}",
            f"{t1}",
            *roi.split(),
        ]
    )
    nonroi2roi = anatdir / "T1_nonroi2roi.mat"
    roi2nonroi = anatdir / "T1_roi2nonroi.mat"
    orig2roi = anatdir / "T1_orig2roi.mat"
    orig2std = anatdir / "T1_orig2std.mat"
    roi2orig = anatdir / "T1_roi2orig.mat"
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{nonroi2roi}",
            "-inverse",
            f"{roi2nonroi}",
        ]
    )
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{orig2roi}",
            "-concat",
            f"{nonroi2roi}",
            f"{orig2std}",
        ]
    )
    subprocess.run(
        [  # noqa: S603
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{roi2orig}",
            "-inverse",
            f"{orig2roi}",
        ]
    )


def _get_high_voxel_mask(t1: Path) -> Path:
    nii: nb.nifti1.Nifti1Image = nb.nifti1.load(t1)
    nonzero_i = np.flatnonzero(nii.get_fdata())
    threshold = np.quantile(nii.get_fdata().flat[nonzero_i], 0.99)
    out = t1.with_name(f"{utils.img_stem(t1)}_lesionmask.nii.gz")
    nb.nifti1.Nifti1Image(
        dataobj=np.asarray(nii.get_fdata() > threshold, dtype=np.int8),
        affine=nii.affine,
        header=nii.header,
    ).to_filename(out)
    return out


def get_t1_from_dir(src_dir: Path, pattern: str) -> Path:
    maybe_t1 = list(d for d in src_dir.rglob(pattern))
    if len(maybe_t1) == 0:
        msg = f"Did not find any {pattern} in {src_dir}"
        raise AssertionError(msg)
    elif len(maybe_t1) > 1:
        msg = f"Found more than one {pattern} in {src_dir}"
        raise AssertionError(msg)
    return maybe_t1[0]


class FSLAnatEntrypoint(tapismpi.TapisMPIEntrypoint):
    precrop: typing.Sequence[bool] | None = None
    mask_high_voxels: typing.Sequence[bool] | None = None
    strongbias: typing.Sequence[bool] | None = None

    @property
    def rank_log(self) -> str:
        return f"fslanat_rank-{self.RANK}.log"

    @property
    def fsldir(self) -> str:
        fsldir = os.getenv("FSLDIR")
        if not isinstance(fsldir, str):
            raise AssertionError("FSLDIR is not set")
        return fsldir

    def get_args(self, outdir: Path) -> list[str]:

        t1 = get_t1_from_dir(outdir, "*T1.nii.gz")
        basename = utils.img_stem(t1)
        anat = _predict_fsl_anat_output(outdir, basename)

        fslflags = []
        if self.precrop:
            fslflags += ["--nocrop", "--noreorient"]
        if self.mask_high_voxels:
            fslflags += ["-m", f"{basename}_lesionmask.nii.gz"]
        if self.strongbias:
            fslflags += ["--strongbias"]
        args = [f"{self.fsldir}/bin/fsl_anat", "-d", str(anat), *fslflags]

        return args

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        out = True
        anatdirs = list((output_dir_to_check / "fslanat").glob("*.anat"))
        if not len(anatdirs) == 1:
            logging.error(
                f"Unexpected number of .anat dirs found in {output_dir_to_check}: {anatdirs=}"
            )
            out = False
        try:
            out &= isinstance(
                fslanat_models.FSLAnatResult.from_root(anatdirs[0]),
                fslanat_models.FSLAnatResult,
            )
        except Exception as e:
            logging.error(e)
            out = False

        return out

    async def prep(self, tmpd_in: Path, tmpd_out: Path) -> None:

        nii = get_t1_from_dir(tmpd_in, "*T1w.nii.gz")
        anat = _predict_fsl_anat_output(tmpd_out, utils.img_stem(nii))
        if not anat.exists():
            utils.mkdir_recursive(anat)

        t1 = shutil.copyfile(nii, anat / "T1.nii.gz")

        if self.precrop:
            _precrop(anat)
        if self.mask_high_voxels:
            _get_high_voxel_mask(t1)

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:

        await self.prep(tmpd_in, tmpd_out / "fslanat")
        async with utils.subprocess_manager(
            log=tmpd_out / self.rank_log,
            args=self.get_args(outdir=tmpd_out / "fslanat"),
        ) as proc:
            await proc.wait()
            if proc.returncode and proc.returncode > 0:
                # remove folder so that archiving detects that there was a failure
                # and sends logs to failure_dst_dir
                if (outdir := tmpd_out / "fslanat").exists():
                    shutil.rmtree(outdir)
                msg = f"fslanat failed with {proc.returncode=}"
                raise RuntimeError(msg)
