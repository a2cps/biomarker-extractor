import logging
import shutil
from pathlib import Path

from biomarkers import datasets, utils
from biomarkers.entrypoints import tapismpi


class CATEntrypoint(tapismpi.TapisMPIEntrypoint):

    def get_args(self, batchfile: Path) -> list[str]:
        return ["/opt/spm/spm12", "batch", str(batchfile)]

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
        nii_out = tmpd_out / "cat12" / nii.name

        logging.info(f"Prepping {nii}")
        nii_out.parent.mkdir()
        shutil.copyfile(nii, nii_out)

        batch = (
            datasets.get_cat_batch()
            .read_text()
            .replace("<UNDEFINED>", f"'{str(nii_out)}'")
        )
        batch_out = nii_out.parent / "batch.m"
        batch_out.write_text(batch)

        return batch_out

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:

        batchfile = await self.prep(tmpd_in, tmpd_out)
        async with utils.subprocess_manager(
            log=tmpd_out / f"cat12_rank-{self.RANK}.log",
            args=self.get_args(batchfile),
        ) as proc:
            await proc.wait()

            # remove extra copies of input files (expect 1 nii, 1 nii.gz)
            for nii in batchfile.parent.glob("*nii*"):
                nii.unlink()

            if proc.returncode and proc.returncode > 0:
                # remove folder so that archiving detects that there was a failure
                # and sends logs to failure_dst_dir
                if batchfile.parent.exists():
                    shutil.rmtree(batchfile.parent)
                msg = f"cat12 failed with {proc.returncode=}"
                raise RuntimeError(msg)
