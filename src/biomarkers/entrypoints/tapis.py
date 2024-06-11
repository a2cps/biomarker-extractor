import logging
import shutil
import socket
import typing
from abc import abstractmethod
from pathlib import Path

import pydantic

from biomarkers import prefect_utils, utils


def configure_basic_logger() -> None:
    host = socket.gethostname()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | %(message)s",
        level=logging.INFO,
    )


def _copy_tapis_files(outdir: Path) -> None:
    # tapis logs tend to be in the form of [jobid].{err,out}
    # this copies them to a destination folder
    for stderr in Path.cwd().glob("*.err"):
        shutil.copyfile(stderr, outdir / stderr.name)
    for stdout in Path.cwd().glob("*.out"):
        shutil.copyfile(stdout, outdir / stdout.name)


class TapisEntrypoint(pydantic.BaseModel):

    ins: typing.Iterable[Path]
    outs: typing.Iterable[Path]
    stage_dir: Path

    @abstractmethod
    def run_flow(self) -> list[Path]:
        raise NotImplementedError

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists()

    def archive(self, src: typing.Sequence[Path]) -> None:
        for o in src:
            logging.info(f"Attempting archive of {o}")
            try:
                dst = o.relative_to(self.stage_dir)
                if self.check_outputs(o):
                    logging.info(f"Copying {o} -> {dst}")
                    if not dst.exists():
                        utils.mkdir_recursive(dst, mode=0o770)
                    shutil.copytree(
                        o,
                        dst,
                        dirs_exist_ok=True,
                        copy_function=shutil.copyfile,
                    )
                    # need one more chmod for after copytree
                    # which preserves permissions of dst itself
                    dst.chmod(0o770)
                else:
                    # in case of failures, it's helpful to keep logs around
                    log_dst = utils.FAILURE_LOG_DST / dst.stem
                    logging.warning(
                        f"Failure detected for {o}. Copying logs to {log_dst}"
                    )
                    if not log_dst.exists():
                        utils.mkdir_recursive(log_dst, mode=0o770)
                    for log in o.glob("*log"):
                        shutil.copyfile(log, log_dst / log.name)
                    _copy_tapis_files(log_dst)
            except Exception as e:
                logging.error(f"Failed to archive {o}: {e}")

    def copy_tapis_logs_to_out(self) -> None:
        for outdir in self.outs:
            if outdir.exists():
                _copy_tapis_files(outdir=outdir)

    async def run(self) -> None:
        if not self.stage_dir.exists():
            self.stage_dir.mkdir(parents=True)
        elif not self.stage_dir.is_dir():
            msg = f"stage_dir must be a directory. found {self.stage_dir}"
            raise AssertionError(msg)

        async with prefect_utils.get_prefect():
            staged_outs = self.run_flow()

        self.archive(staged_outs)

        # copy tapis logs at the end because archive will add more lines
        self.copy_tapis_logs_to_out()
