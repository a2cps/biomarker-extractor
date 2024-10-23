import logging
import shutil
import tempfile
from pathlib import Path

import nibabel as nb
from nibabel import processing

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


def get_gift_args(bidsdir: Path, derivative_name: str, config: Path) -> list[str]:
    # https://github.com/trendscenter/gift-bids/blob/b176aa119e55a63c557fc3a0d164809fac14e6cb/Dockerfile
    deriv = bidsdir / "derivatives" / "gift" / "derivatives" / derivative_name
    args = [
        "/app/run.sh",
        str(bidsdir),
        str(deriv),
        "participant",
        "--skip-bids-validator",
        "--config",
        str(config),
    ]
    utils.mkdir_recursive(deriv, mode=utils.DIR_PERMISSIONS)
    return args


# need to force gift to run one just one run at a time (no collapsing)
# this means that we need temporary folders with just a single
# func scan


def make_1_run_bids(src: Path, dst: Path, bold: Path) -> None:
    sub = utils.get_sub(bold)
    ses = utils.get_ses(bold)
    func = dst / f"sub-{sub}" / f"ses-{ses}" / "func"
    utils.mkdir_recursive(func, mode=utils.DIR_PERMISSIONS)
    # dst won't exist yet, so wait until the above mkdir before
    # copying files
    shutil.copyfile(src / "dataset_description.json", dst / "dataset_description.json")
    shutil.copyfile(bold, func / bold.name)
    anat = dst / f"sub-{sub}" / f"ses-{ses}" / "anat"
    utils.mkdir_recursive(anat, mode=utils.DIR_PERMISSIONS)
    for anatf in (bold.parent.parent / "anat").glob("*gz"):
        shutil.copyfile(anatf, anat / anatf.name)


class GIFTEntrypoint(tapismpi.TapisMPIEntrypoint):
    configs: dict[str, Path]
    voxel_size: float = 2.4
    smooth_fwhm: float = 6.0

    async def do_single_run_bids(self, in_dir: Path, out_dir: Path, bold: Path) -> int:
        returncode = 0
        for config_label, config in self.configs.items():
            with tempfile.TemporaryDirectory() as _tmpd:
                tmpd = Path(_tmpd)
                make_1_run_bids(in_dir, tmpd, bold)

                async with utils.subprocess_manager(
                    log=out_dir
                    / f"gift_rank-{self.RANK}_{utils.img_stem(bold)}_{config_label}.log",
                    args=get_gift_args(
                        bidsdir=tmpd,
                        derivative_name=f"{utils.img_stem(bold)}_{config_label}",
                        config=config,
                    ),
                ) as proc:
                    await proc.wait()
                    if proc.returncode is None:
                        returncode += 1
                        break
                    elif proc.returncode > 0:
                        returncode += proc.returncode
                        break

                gift_dir = out_dir / "gift"
                shutil.copytree(
                    tmpd,
                    gift_dir,
                    dirs_exist_ok=True,
                    copy_function=shutil.copyfile,
                )
                gift_dir.chmod(utils.DIR_PERMISSIONS)
                # gift offers no control over the creation of the folder
                # derivatives/gift, nor the file names. Since we're running
                # multiple model orders, we need to ensure that the
                # files within this folder are not overwritten with a second
                # config
                dst = gift_dir / "derivatives" / f"gift-{config_label}"
                src = gift_dir / "derivatives" / "gift"
                if dst.exists():
                    shutil.copytree(
                        src,
                        dst,
                        dirs_exist_ok=True,
                        copy_function=shutil.copyfile,
                    )
                    shutil.rmtree(src)
                else:
                    src.rename(dst)

        return returncode

    def prep(self, to_prep: Path):
        for bold in to_prep.rglob("*MNI*bold.nii.gz"):
            nii = nb.nifti1.load(bold)
            logging.info(f"resampling {bold}")
            resampled = nb.funcs.concat_images(
                [
                    processing.resample_to_output(
                        nii.slicer[:, :, :, vol], voxel_sizes=self.voxel_size
                    )
                    for vol in range(nii.shape[-1])
                ]
            )
            processing.smooth_image(resampled, self.smooth_fwhm).to_filename(bold)

        # loop involves renaming so must generator to list
        for bold in list(to_prep.rglob("*MNI*")):
            bold.rename(
                bold.with_name(
                    bold.name.replace("space-MNI152NLin2009cAsym_desc-preproc_", "")
                )
            )
        # end up with a few extra files that need deleting
        for bold in list(to_prep.rglob("*")):
            if "desc-" in bold.name:
                bold.unlink()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> int:
        returncode = 0
        self.prep(in_dir)
        for bold in in_dir.rglob("*bold.nii.gz"):
            logging.info(f"running gift on {bold}")
            returncode += await self.do_single_run_bids(
                in_dir=in_dir, out_dir=out_dir, bold=bold
            )
            logging.info(f"{bold} done")
        return returncode
