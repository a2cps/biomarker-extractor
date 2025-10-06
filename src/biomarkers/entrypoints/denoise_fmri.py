import logging
from pathlib import Path

from biomarkers.entrypoints import tapismpi
from biomarkers.flows import denoise_fmri


class DenoiseFMRIEntrypoint(tapismpi.TapisMPIEntrypoint):
    high_pass: float | None = 0.01
    n_non_steady_state_tr: int = 15
    detrend: bool = True
    fwhm: float | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        await denoise_fmri.denoise_flow(in_dir, out=out_dir)
        logging.info("done")
