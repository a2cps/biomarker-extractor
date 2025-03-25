import typing
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


class BEDPOSTXEntrypoint(tapismpi.TapisMPIEntrypoint):
    participant_label: typing.Sequence[str]
    ses_label: typing.Sequence[str]

    def get_args(self, qsirecondir: Path, outdir: Path) -> list[str]:
        return [
            "bedpostx_wrapper",
            str(self.participant_label[self.RANK]),
            str(self.ses_label[self.RANK]),
            str(qsirecondir),
            str(outdir),
        ]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return (output_dir_to_check / "bedpostx").exists()

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        async with utils.subprocess_manager(
            log=tmpd_out / f"bedpostx_rank-{self.RANK}.log",
            args=self.get_args(tmpd_in, tmpd_out),
        ) as proc:
            await proc.wait()

            if proc.returncode and proc.returncode > 0:
                msg = f"bedpostx failed with {proc.returncode=}"
                raise RuntimeError(msg)
