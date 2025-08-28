import logging
import shutil
import tempfile
from pathlib import Path

from nilearn import image

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


def get_template_from_config(config: Path) -> Path:
    lines = config.read_text().splitlines()
    for line in lines:
        if "refFiles" in line:
            reffile = Path(
                line.removeprefix("refFiles")
                .replace("=", "")
                .removesuffix(";")
                .replace("'", "")
            )
            if not reffile.exists():
                msg = "refFiles extracted but does not exist"
                raise RuntimeError(msg)

            return reffile

    msg = "refFiles not found"
    raise RuntimeError(msg)


def make_1_run_bids(src: Path, dst: Path, bold: Path) -> None:
    # need to force gift to run one just one run at a time (no collapsing)
    # this means that we need temporary folders with just a single
    # func scan
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
    smooth_fwhm: float = 6.0
    voxel_size: float = 2.0
    low_pass: float = 0.15
    template: Path = Path("/opt/gift/Neuromark_fMRI_2.1_modelorder-multi.nii")

    _ref: Path | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "gift").exists()

    async def do_single_run_bids(self, in_dir: Path, out_dir: Path, bold: Path) -> None:
        gift_dir = out_dir / "gift"
        for config_label, config in self.configs.items():
            with tempfile.TemporaryDirectory() as _tmpd:
                tmpd = Path(_tmpd)
                make_1_run_bids(in_dir, tmpd, bold)

                async with utils.subprocess_manager(
                    log=out_dir
                    / f"gift_rank-{self.RANK}_{utils.img_stem(bold)}_{config_label}.log",
                    args=self.get_args(
                        bidsdir=tmpd,
                        derivative_name=f"{utils.img_stem(bold)}_{config_label}",
                        config=config,
                    ),
                ) as proc:
                    await proc.wait()
                    if proc.returncode is None or proc.returncode > 0:
                        # remove folder so that archiving detects that there was a failure
                        # and sends logs to failure_dst_dir
                        if gift_dir.exists():
                            shutil.rmtree(gift_dir)
                        msg = f"gift failed with {proc.returncode=}"
                        raise RuntimeError(msg)

                shutil.copytree(
                    tmpd, gift_dir, dirs_exist_ok=True, copy_function=shutil.copyfile
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

    def prep(self, to_prep: Path):
        for bold in list(to_prep.rglob("*MNI*bold.nii.gz")):
            logging.info(f"resampling {bold} to template")
            resampled = image.resample_to_img(
                bold, self.template, force_resample=True, copy_header=True
            )

            logging.info(f"smoothing and cleaning {bold}")
            image.clean_img(
                imgs=image.smooth_img(resampled, fwhm=self.smooth_fwhm),
                t_r=utils.get_tr(resampled),
                low_pass=self.low_pass,
                detrend=True,
                standardize=False,
                clean__extrapolate=False,
            ).to_filename(bold)

        # GIFT fails to recognize space-* files as relevant, so need to simplify names
        # loop involves renaming so must generator to list
        for bold in list(to_prep.rglob("*MNI*")):
            if "res-" in bold.name:
                dst = bold.with_name(
                    bold.name.replace(
                        "space-MNI152NLin2009cAsym_res-2_desc-preproc_", ""
                    )
                )
            else:
                dst = bold.with_name(
                    bold.name.replace("space-MNI152NLin2009cAsym_desc-preproc_", "")
                )
            logging.info(f"renaming {bold} -> {dst}")
            bold.rename(dst)

    def tidy(self, out_dir: Path) -> None:
        # unlinking files after gzip, so convert to list
        # before iterating over
        for nii in list(out_dir.rglob("*nii")):
            utils.gzip_file(nii, nii.with_suffix(".nii.gz"))
            nii.unlink()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        self.prep(in_dir)
        for bold in in_dir.rglob("*bold.nii.gz"):
            logging.info(f"running gift on {bold}")
            await self.do_single_run_bids(in_dir=in_dir, out_dir=out_dir, bold=bold)
            logging.info(f"{bold} done")

        logging.info(f"compressing niis in {out_dir}")
        self.tidy(out_dir=out_dir)

    @staticmethod
    def get_args(bidsdir: Path, derivative_name: str, config: Path) -> list[str]:
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
