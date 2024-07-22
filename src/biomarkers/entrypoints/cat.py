import logging
import shutil
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


class CATEntrypoint(tapismpi.TapisMPIEntrypoint):
    batchfile: Path

    def get_args(self, nii: Path) -> list[str]:
        return [
            "/usr/local/bin/cat_standalone.sh",
            str(nii),
            "-b",
            str(self.batchfile),
        ]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        out = True
        report = list((output_dir_to_check).rglob("*pdf"))
        if not len(report) == 1:
            logging.error(
                f"Unexpected number of reports found in {output_dir_to_check}: {report}"
            )
            out = False

        return out

    async def prep(self, tmpd_in: Path, tmpd_out: Path) -> Path:

        logging.info(f"Looking for *T1w.nii.gz in {tmpd_in}")
        maybe_nii = list(d for d in tmpd_in.rglob("*T1w.nii.gz"))
        if len(maybe_nii) == 0:
            msg = f"Did not find any *T1w.nii.gz in {tmpd_in}"
            raise AssertionError(msg)
        elif len(maybe_nii) > 1:
            logging.warning(
                f"Unexpected number of  *T1w.nii.gz found in {tmpd_in}: {maybe_nii}. Taking first."
            )
        nii = maybe_nii[0]
        nii_out = tmpd_out / "cat" / nii.name

        logging.info(f"Prepping {nii}")
        nii_out.parent.mkdir()
        shutil.copyfile(nii, nii_out)

        return nii_out

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:

        staged_in = await self.prep(tmpd_in, tmpd_out)
        async with utils.subprocess_manager(
            log=tmpd_out / f"cat12_rank-{self.RANK}.log",
            args=self.get_args(nii=staged_in),
        ) as proc:
            await proc.wait()
            staged_in.unlink()
            (staged_in.parent / staged_in.name.removesuffix(".gz")).unlink()
