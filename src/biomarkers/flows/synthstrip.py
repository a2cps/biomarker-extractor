import shutil
import subprocess
from pathlib import Path

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
    args = ["mri_synthstrip", "-i", str(src), "-m", str(mask), "-t", "1"]
    if not csf:
        args.append("--no-csf")
    return args


async def synthstrip_flow(nii: Path, out_dir: Path) -> None:
    for csf in [True, False]:
        async with utils.subprocess_manager(
            log=subprocess.DEVNULL, args=get_args(src=nii, dst=out_dir, csf=csf)
        ) as proc:
            await proc.wait()
            if proc.returncode and proc.returncode > 0:
                # remove folder so that archiving detects that there was a failure
                # and sends logs to failure_dst_dir
                if (outdir := out_dir / "synthstrip").exists():
                    shutil.rmtree(outdir)
                msg = f"synthstrip failed with {proc.returncode=}"
                raise RuntimeError(msg)
