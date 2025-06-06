import logging
from pathlib import Path

from biomarkers.entrypoints import tapismpi
from biomarkers.flows import postgift


class PostGIFTEntrypoint(tapismpi.TapisMPIEntrypoint):
    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "biomarkers").exists()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        logging.info(f"extracting gift measures from {in_dir}")
        postgift.postgift_flow(indir=in_dir, outdir=out_dir)
