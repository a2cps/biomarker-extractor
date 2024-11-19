import logging
from pathlib import Path

from biomarkers import imgs
from biomarkers.entrypoints import tapismpi
from biomarkers.flows import signature
from biomarkers.models import fmriprep


class SignatureEntrypoint(tapismpi.TapisMPIEntrypoint):
    high_pass: float | None = None
    low_pass: float | None = 0.1
    n_non_steady_state_tr: int = 12
    detrend: bool = True
    fwhm: float | None = None
    winsorize: bool = True
    space: fmriprep.SPACE = "MNI152NLin6Asym"
    compcor_label: imgs.COMPCOR_LABEL | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return all(
            [
                (output_dir_to_check / d).exists()
                for d in (
                    "signature-by-part",
                    "signature-by-run",
                    "signature-by-tr",
                    "signature-cleaned",
                    "signature-confounds",
                    "signature-labels",
                )
            ]
        )

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
            compcor_label=self.compcor_label,
        )
        logging.info("done")
