import tempfile
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


def extend_arg(
    args: list[str], name: str, value: str | int | bool | Path | None = None
):
    if value is not None:
        args.extend([name, str(value)])


class MRIQCEntrypoint(tapismpi.TapisMPIEntrypoint):
    n_workers: int | None = None
    n_threads: int | None = None
    mem_gb: int | None = None

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list((output_dir_to_check / "mriqc").glob("*html"))) > 0
        )

    def get_args(
        self, bidsdir: Path, outdir: Path, work_dir: Path
    ) -> list[str]:
        args = ["mriqc", "--notrack", "--no-sub"]
        to_extend = {
            "-n-cpus": self.n_workers,
            "--omp-nthreads": self.n_threads,
            "--mem-gb": self.mem_gb,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        args.extend([str(bidsdir), str(outdir / "mriqc"), "participant"])

        return args

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:

        with tempfile.TemporaryDirectory() as tmpd:
            async with utils.subprocess_manager(
                log=tmpd_out / f"mriqc_rank-{self.RANK}.log",
                args=self.get_args(
                    bidsdir=tmpd_in, outdir=tmpd_out, work_dir=Path(tmpd)
                ),
            ) as proc:
                await proc.wait()
                if proc.returncode and proc.returncode > 0:
                    msg = f"mriqc failed with {proc.returncode=}"
                    raise RuntimeError(msg)
