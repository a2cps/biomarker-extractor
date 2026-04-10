import shutil
from pathlib import Path

import nibabel as nb
import polars as pl

from biomarkers import utils


def get_args(src: Path, dst: Path, csf: bool = True) -> list[str]:
    if csf:
        desc = "braincsf"
    else:
        desc = "brain"
    sub = utils.get_sub(src)
    ses = utils.get_ses(src)
    mask = (
        dst
        / "synthstrip"
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "anat"
        / src.name.replace("T1w.nii.gz", f"desc-{desc}_mask.nii.gz")
    )
    mask.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "python",
        "/opt/synthstrip/mri_synthstrip.py",
        "-i",
        str(src),
        "-m",
        str(mask),
        "-t",
        "1",
    ]
    if not csf:
        args.extend(["--model", "/opt/synthstrip/synthstrip.nocsf.1.pt"])
    else:
        args.extend(["--model", "/opt/synthstrip/synthstrip.1.pt"])
    return args


async def synthstrip_flow(nii: Path, out_dir: Path) -> None:
    for csf in [True, False]:
        async with utils.subprocess_manager(
            log=Path("/dev/null"), args=get_args(src=nii, dst=out_dir, csf=csf)
        ) as proc:
            await proc.wait()
            if proc.returncode and proc.returncode > 0:
                # remove folder so that archiving detects that there was a failure
                # and sends logs to failure_dst_dir
                if (outdir := out_dir / "synthstrip").exists():
                    shutil.rmtree(outdir)
                msg = f"synthstrip failed with {proc.returncode=}"
                raise RuntimeError(msg)

    volumes = {}
    for f in out_dir.rglob("*nii.gz"):
        volumes[f.stem] = nb.nifti1.Nifti1Image.load(f).get_fdata().sum().astype(int)

    pl.DataFrame(volumes).unpivot(
        variable_name="src", value_name="n_voxels"
    ).with_columns(
        sub=pl.col("src").str.extract(r"sub-(\w+)_"),
        ses=pl.col("src").str.extract(r"ses-(\w+)_"),
        csf=pl.col("src").str.contains("csf"),
    ).drop("src").write_csv(out_dir / "synthstrip" / "volumes.tsv", separator="\t")
