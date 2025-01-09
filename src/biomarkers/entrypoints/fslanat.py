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


def run_and_log_stdout(cmd: list[str], log: Path) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(out.stdout)
    return out.stdout


def get_lesionmask(anat: Path) -> Path:
    return anat / "T1_lesionmask.nii.gz"


def _reorient2standard(t1: Path) -> None:
    """
    $FSLDIR/bin/fslmaths ${T1} ${T1}_orig
    $FSLDIR/bin/fslreorient2std ${T1} > ${T1}_orig2std.mat
    $FSLDIR/bin/convert_xfm -omat ${T1}_std2orig.mat -inverse ${T1}_orig2std.mat
    $FSLDIR/bin/fslreorient2std ${T1} ${T1}
    """
    t1_std2orig = t1.with_name("T1_std2orig.mat")
    t1_orig2std = t1.with_name("T1_orig2std.mat")
    subprocess.run(
        [f"{os.getenv('FSLDIR')}/bin/fslmaths", f"{t1}", f"{t1.with_name('T1_orig')}"]
    )
    run_and_log_stdout(
        [f"{os.getenv('FSLDIR')}/bin/fslreorient2std", f"{t1}"], log=t1_orig2std
    )
    subprocess.run(
        [
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{t1_std2orig}",
            "-inverse",
            f"{t1_orig2std}",
        ]
    )
    subprocess.run([f"{os.getenv('FSLDIR')}/bin/fslreorient2std", f"{t1}", f"{t1}"])


def _precrop(anat: Path):
    """
    im=failing
    outputname=out
    i90=$( fslstats $im -P 90 )
    fslmaths $im -mul 0 -randn -mul $( echo "$i90 * 0.005" | bc -l ) -mas $im -add $im ${im}_noisy
    roivals=$( robustfov -i ${im}_noisy | grep 0 )
    fslroi $im $outputname $roivals
    """
    t1 = anat / "T1.nii.gz"

    # REORIENTATION 2 STANDARD
    _reorient2standard(t1)

    # AUTOMATIC CROPPING
    fullfov = t1.with_name("T1_fullfov.nii.gz")
    shutil.move(t1, fullfov)
    nii = nb.nifti1.Nifti1Image.load(fullfov)
    nonzero_i = np.flatnonzero(nii.get_fdata())
    sigma = np.quantile(nii.get_fdata().flat[nonzero_i], 0.9) * 0.005
    noisy: np.ndarray = nii.get_fdata().copy()
    noisy.flat[nonzero_i] += np.random.normal(0, sigma, size=nonzero_i.shape)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmpfile:
        nb.ni1.Nifti1Image(
            dataobj=noisy, affine=nii.affine, header=nii.header
        ).to_filename(tmpfile.name)

        log = subprocess.run(
            [
                f"{os.getenv('FSLDIR')}/bin/robustfov",
                "-m",
                f"{anat / 'T1_roi2nonroi.mat'}",
                "-i",
                f"{tmpfile.name}",
            ],
            capture_output=True,
            text=True,
        )
    roi = log.stdout.splitlines()[1]
    (anat / "T1_roi.log").write_text(roi)

    subprocess.run(
        [f"{os.getenv('FSLDIR')}/bin/fslroi", f"{fullfov}", f"{t1}", *roi.split()]
    )
    nonroi2roi = anat / "T1_nonroi2roi.mat"
    roi2nonroi = anat / "T1_roi2nonroi.mat"
    orig2roi = anat / "T1_orig2roi.mat"
    orig2std = anat / "T1_orig2std.mat"
    roi2orig = anat / "T1_roi2orig.mat"
    subprocess.run(
        [
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{nonroi2roi}",
            "-inverse",
            f"{roi2nonroi}",
        ]
    )
    subprocess.run(
        [
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{orig2roi}",
            "-concat",
            f"{nonroi2roi}",
            f"{orig2std}",
        ]
    )
    subprocess.run(
        [
            f"{os.getenv('FSLDIR')}/bin/convert_xfm",
            "-omat",
            f"{roi2orig}",
            "-inverse",
            f"{orig2roi}",
        ]
    )


def _get_high_voxel_mask(anat: Path) -> Path:
    t1 = anat / "T1.nii.gz"
    nii: nb.nifti1.Nifti1Image = nb.nifti1.Nifti1Image.load(t1)
    nonzero_i = np.flatnonzero(nii.get_fdata())
    threshold = np.quantile(nii.get_fdata().flat[nonzero_i], 0.99)
    out = get_lesionmask(anat)
    nb.nifti1.Nifti1Image(
        dataobj=np.asarray(nii.get_fdata() > threshold, dtype=np.int8),
        affine=nii.affine,
        header=nii.header,
    ).to_filename(out)
    return out


class FSLAnatEntrypoint(tapismpi.TapisMPIEntrypoint):
    precrop: typing.Sequence[bool] | None = None
    mask_high_voxels: typing.Sequence[bool] | None = None
    strongbias: typing.Sequence[bool] | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        out = True
        anats = list(output_dir_to_check.glob("*.anat"))
        if not len(anats) == 1:
            logging.error(
                f"Unexpected number of .anat dirs found in \
                    {output_dir_to_check}: {anats=}"
            )
            out = False
        try:
            out &= isinstance(
                fslanat_models.FSLAnatResult.from_root(anats[0]),
                fslanat_models.FSLAnatResult,
            )
        except Exception as e:
            logging.error(e)
            out = False

        return out

    def prep(self, tmpd_in: Path, tmpd_out: Path) -> Path:
        logging.info(f"Looking for *T1w.nii.gz in {tmpd_in}")
        maybe_nii = list(d for d in tmpd_in.rglob("*T1w.nii.gz"))
        if len(maybe_nii) == 0:
            msg = f"Did not find any *T1w.nii.gz in {tmpd_in}"
            raise ValueError(msg)
        elif len(maybe_nii) > 1:
            logging.warning(
                f"Unexpected number of  *T1w.nii.gz found in \
                    {tmpd_in}: {maybe_nii}. Taking first."
            )
        nii = maybe_nii[0]

        basename = utils.img_stem(nii)
        anat = _predict_fsl_anat_output(tmpd_out, basename)

        utils.mkdir_recursive(anat)

        shutil.copyfile(nii, anat / "T1.nii.gz")

        if self.precrop and self.precrop[self.RANK]:
            _precrop(anat)
        if self.mask_high_voxels and self.mask_high_voxels[self.RANK]:
            _get_high_voxel_mask(anat)

        return anat

    def get_args(self, anat: Path) -> list[str]:
        fslflags = []
        if self.precrop and self.precrop[self.RANK]:
            fslflags += ["--nocrop", "--noreorient"]
        if self.mask_high_voxels and self.mask_high_voxels[self.RANK]:
            fslflags += ["-m", str(get_lesionmask(anat))]
        if self.strongbias and self.strongbias[self.RANK]:
            fslflags += ["--strongbias"]
        return [f"{os.getenv('FSLDIR')}/bin/fsl_anat", "-d", str(anat), *fslflags]

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        staged_in = self.prep(tmpd_in, tmpd_out)
        if staged_in:
            async with utils.subprocess_manager(
                log=tmpd_out / f"fslanat_rank-{self.RANK}.log",
                args=self.get_args(anat=staged_in),
            ) as proc:
                await proc.wait()
