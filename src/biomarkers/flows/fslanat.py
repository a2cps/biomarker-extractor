import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import prefect
import prefect.logging

from biomarkers import utils


def _predict_fsl_anat_output(out: Path, basename: str) -> Path:
    return (out / basename).with_suffix(".anat").absolute()


def run_and_log_stdout(cmd: list[str], log: Path) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    log.write_text(out.stdout)
    return out.stdout


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
    run_and_log_stdout(
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


@prefect.task
async def _fslanat(
    image: Path,
    out: Path,
    precrop: bool = False,  # noqa: FBT002, FBT001
    strongbias: bool = False,  # noqa: FBT002, FBT001
    mask_high_voxels: bool = False,
):
    basename = utils.img_stem(image)
    anat = _predict_fsl_anat_output(out, basename)
    logger = prefect.get_run_logger()

    if not anat.exists():
        utils.mkdir_recursive(anat)

    shutil.copyfile(image, anat / "T1.nii.gz")

    fslflags = []
    if precrop:
        _precrop(anat)
        fslflags += ["--nocrop", "--noreorient"]
    if mask_high_voxels:
        mask = _get_high_voxel_mask(image)
        fslflags += ["-m", str(mask)]
    if strongbias:
        fslflags += ["--strongbias"]
    args = [
        f"{os.getenv('FSLDIR')}/bin/fsl_anat",
        "-d",
        str(anat),
        *fslflags,
    ]
    logger.info(args)

    async with utils.subprocess_manager(
        log=out / f"fslanat-{basename}.log", args=args
    ) as proc:
        await proc.wait()


@prefect.flow
def fslanat_flow(
    images: typing.Sequence[Path],
    outdirs: typing.Sequence[Path],
    precrops: typing.Sequence[bool] | None = None,
    strongbias: typing.Sequence[bool] | None = None,
    mask_high_voxels: typing.Sequence[bool] | None = None,
) -> None:
    _precrops = utils.compare_arg_lengths(
        precrops, images, ("precrops", "images")
    )
    _strongbias = utils.compare_arg_lengths(
        strongbias, images, ("strongbias", "images")
    )
    _mask_high_voxels = utils.compare_arg_lengths(
        mask_high_voxels, images, ("mask_high_voxels", "images")
    )

    for image, precrop, strongb, mask, out in zip(
        images, _precrops, _strongbias, _mask_high_voxels, outdirs, strict=True
    ):
        _fslanat.submit(
            image,
            out=out,
            precrop=precrop,
            strongbias=strongb,
            mask_high_voxels=mask,
        )
