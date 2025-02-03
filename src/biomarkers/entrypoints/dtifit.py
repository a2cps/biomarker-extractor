import typing
from pathlib import Path

from biomarkers import datasets, utils
from biomarkers.entrypoints import tapismpi


class DTIFitEntrypoint(tapismpi.TapisMPIEntrypoint):
    fs_license_file: Path
    participant_label: typing.Sequence[str]
    ses_label: typing.Sequence[str]

    def get_args(self, qsiprepdir: Path, outdir: Path) -> list[str]:
        return [
            str(datasets.get_dtifit_script()),
            str(qsiprepdir),
            str(outdir),
            str(self.fs_license_file),
            str(self.participant_label[self.RANK]),
            str(self.ses_label[self.RANK]),
        ]

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return all(
            (output_dir_to_check / d).exists()
            for d in [
                "qsirecon/derivatives/qsirecon-FSL",
                "split_shells",
                "dtifit-multishell",
                "dtifit-b1000",
                "dtifit-b2000",
                "dtifit-b3000",
            ]
        )

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        async with utils.subprocess_manager(
            log=tmpd_out / f"dtifit_rank-{self.RANK}.log",
            args=self.get_args(tmpd_in, tmpd_out),
        ) as proc:
            await proc.wait()

            if proc.returncode and proc.returncode > 0:
                msg = f"dtifit failed with {proc.returncode=}"
                raise RuntimeError(msg)
