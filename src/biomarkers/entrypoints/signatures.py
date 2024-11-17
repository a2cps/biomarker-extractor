import logging
from pathlib import Path

from biomarkers.entrypoints import tapismpi
from biomarkers.flows import signature


class SignatureEntrypoint(tapismpi.TapisMPIEntrypoint):
    high_pass: float | None = None
    low_pass: float | None = 0.1
    n_non_steady_state_tr: int = 12
    detrend: bool = True
    fwhm: float | None = None
    winsorize: bool = True
    space: str = "MNI152NLin2009cAsym"

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "signatures").exists()

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        logging.info(f"extracting signatures from {in_dir}")
        signature.signature_flow(
            in_dir,
            out=out_dir,
            high_pass=self.high_pass,
            low_pass=self.low_pass,
            n_non_steady_state_tr=self.n_non_steady_state_tr,
            detrend=self.detrend,
            fwhm=self.fwhm,
            winsorize=self.winsorize,
            space=self.space,
        )
        logging.info("done")
