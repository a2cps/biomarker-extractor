import logging
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.flows import postgift


class PostGIFTEntrypoint(tapismpi.TapisMPIEntrypoint):
    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "biomarkers").exists()

    def prep(self, tmpd_in: Path) -> None:
        logging.info(f"Fixing filenames for {tmpd_in=}")
        # there are both .nii files and .gz files (that is, not .nii.gz)
        for f in tmpd_in.rglob("*gz"):
            if "nii" not in f.name:
                f.rename(f.with_name(f.name.replace(".gz", ".nii.gz")))

        for nii in list(tmpd_in.rglob("*nii")):
            utils.gzip_file(nii, nii.parent / f"{nii.name}.gz")
            nii.unlink()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        self.prep(tmpd_in=in_dir)
        logging.info(f"extracting gift measures from {in_dir}")
        postgift.postgift_flow(indir=in_dir, outdir=out_dir)
