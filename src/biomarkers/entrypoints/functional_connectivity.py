import logging
from pathlib import Path

from biomarkers import imgs
from biomarkers.entrypoints import tapismpi
from biomarkers.flows import functional_connectivity
from biomarkers.models import fmriprep


class FunctionalConnectivityEntrypoint(tapismpi.TapisMPIEntrypoint):
    high_pass: float | None = 0.01
    low_pass: float | None = 0.1
    n_non_steady_state_tr: int = 15
    detrend: bool = True
    fwhm: float | None = None
    winsorize: bool = True
    space: fmriprep.SPACE = "MNI152NLin2009cAsym"
    compcor_label: imgs.COMPCOR_LABEL | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return all(
            [
                (output_dir_to_check / d).exists()
                for d in ("connectivity", "cleaned", "confounds")
            ]
        )

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        logging.info(f"extracting functional connectivity from {in_dir}")
        functional_connectivity.connectivity_flow(
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
