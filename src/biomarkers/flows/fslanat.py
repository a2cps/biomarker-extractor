import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import prefect

from biomarkers import utils
from biomarkers.models.fslanat import FSLAnatResult


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


@prefect.task
def _fslanat(
    image: Path,
    out: Path,
    precrop: bool = False,  # noqa: FBT002, FBT001
    strongbias: bool = False,  # noqa: FBT002, FBT001
):
    basename = utils.img_stem(image)
    anat = _predict_fsl_anat_output(out, basename)

    # if the output already exists, we don't want this to run again.
    # fsl_anat automatically and always adds .anat to the value of -o, so we check for
    # the existence of that predicted output, but then feed in the unmodified value of
    # -o to the task
    if not anat.exists():
        with tempfile.TemporaryDirectory(suffix=".anat") as _tmpdir:
            tmpdir = Path(_tmpdir)
            # (tmpdir / "T1.nii.gz").symlink_to(image)
            shutil.copy2(image, tmpdir / "T1.nii.gz")
            fslflags = []
            if precrop:
                _precrop(tmpdir)
                fslflags += ["--nocrop", "--noreorient"]
            if strongbias:
                fslflags += ["--strongbias"]

            subprocess.run(
                [  # noqa: S603
                    f"{os.getenv('FSLDIR')}/bin/fsl_anat",
                    "-d",
                    f"{tmpdir}",
                    *fslflags,
                ],
                capture_output=True,
            )
            # test that all expected outputs exist
            FSLAnatResult.from_root(tmpdir)

            # store in final destination
            shutil.copytree(tmpdir, anat)


@prefect.flow
def fslanat_flow(
    images: typing.Sequence[Path],
    out: Path,
    precrops: typing.Sequence[bool] | None = None,
    strongbias: typing.Sequence[bool] | None = None,
) -> None:
    if precrops is None:
        _precrops = [False] * len(images)
    else:
        if not len(images) == len(precrops):
            msg = f"""
            If precrops is provided, it must have the same lengths as images.
            Found {len(images)=} and {len(precrops)=}
            """
            raise AssertionError(msg)
        _precrops = precrops
    if strongbias is None:
        _strongbias = [False] * len(images)
    else:
        if not len(images) == len(strongbias):
            msg = f"""
            If strongbias is provided, it must have the same lengths as images.
            Found {len(images)=} and {len(strongbias)=}
            """
            raise AssertionError(msg)
        _strongbias = strongbias

    for image, precrop, strongb in zip(
        images, _precrops, _strongbias, strict=True
    ):
        _fslanat.submit(image, out=out, precrop=precrop, strongbias=strongb)
