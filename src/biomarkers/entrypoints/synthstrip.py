import logging
from pathlib import Path

from biomarkers.entrypoints import tapismpi
from biomarkers.flows import synthstrip


class SynthStripEntrypoint(tapismpi.TapisMPIEntrypoint):
    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list(output_dir_to_check.rglob("**nii.gz"))) == 2
        )

    async def prep(self, tmpd_in: Path) -> Path:
        maybe_nii = list(d for d in tmpd_in.rglob("*T1w.nii.gz"))
        if len(maybe_nii) == 0:
            msg = f"Did not find any *T1w.nii.gz in {tmpd_in}"
            raise AssertionError(msg)
        elif len(maybe_nii) > 1:
            logging.warning(
                f"Unexpected number of *T1w.nii.gz found in {tmpd_in}: {maybe_nii}. Taking first."
            )
        nii = maybe_nii[0]
        logging.info(f"Will generate masks for {nii}")
        return nii

    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        nii = await self.prep(in_dir)
        await synthstrip.synthstrip_flow(nii, out_dir=out_dir)
