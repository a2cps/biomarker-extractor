import asyncio
import logging
import shutil
import socket
import tempfile
import typing
from abc import abstractmethod
from pathlib import Path

import pydantic
from mpi4py import MPI

from biomarkers import utils
from biomarkers.entrypoints import tapis


def configure_mpi_logger() -> None:
    host = socket.gethostname()
    rank = MPI.COMM_WORLD.Get_rank()
    usize = MPI.COMM_WORLD.Get_size()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | {rank=} | {usize=} | %(message)s",
        level=logging.INFO,
    )


class TapisMPIEntrypoint(pydantic.BaseModel):

    ins: typing.Sequence[pydantic.DirectoryPath]
    outs: typing.Sequence[Path]
    timeout: int | float | None = None
    stage_ignore_patterns: (
        None | typing.Callable[[str, list[str]], typing.Iterable[str]]
    ) = None
    RANK: int = pydantic.Field(default_factory=MPI.COMM_WORLD.Get_rank)
    USIZE: int = pydantic.Field(default_factory=MPI.COMM_WORLD.Get_size)

    @abstractmethod
    async def run_flow(self, in_dir: Path, out_dir: Path) -> int:
        raise NotImplementedError

    def stage(self, dst: Path) -> Path:
        for rank, src in enumerate(self.ins):
            if rank == self.RANK:
                logging.info(f"Staging files for {src=} -> {dst=}")
                shutil.copytree(
                    src,
                    dst,
                    ignore=self.stage_ignore_patterns,
                    dirs_exist_ok=True,
                )
            # ensure that only one copy happens at a time
            MPI.COMM_WORLD.barrier()
        return dst

    def archive(self, src: Path, returncode: int | None) -> None:
        for rank, dst in enumerate(self.outs):
            try:
                if rank == self.RANK:
                    if returncode == 0:
                        logging.info(f"Copying {src} -> {dst}")
                        if not dst.exists():
                            utils.mkdir_recursive(dst, mode=0o770)
                        shutil.copytree(
                            src,
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
                            f"Failure detected for {self.outs[self.RANK]=}. Copying logs to {log_dst}"
                        )
                        if not log_dst.exists():
                            utils.mkdir_recursive(log_dst, mode=0o770)
                        for log in src.glob("*log"):
                            shutil.copyfile(log, log_dst / log.name)
                        tapis._copy_tapis_files(log_dst)
                # ensure that only one copy happens at a time
            except Exception as e:
                logging.error(
                    f"Failed to archive {self.outs[self.RANK]=}: {e}"
                )
            finally:
                MPI.COMM_WORLD.barrier()

    async def run(self):
        with tempfile.TemporaryDirectory() as _tmpd_in:
            tmpd_in = self.stage(dst=Path(_tmpd_in))
            with tempfile.TemporaryDirectory() as _tmpd_out:
                tmpd_out = Path(_tmpd_out)
                try:
                    # jobs could get stuck for one process, which would prevent other tasks
                    # from archiving outputs. This forces an error if things have been
                    # going for too long
                    returncode = await asyncio.wait_for(
                        self.run_flow(tmpd_in, tmpd_out), timeout=self.timeout
                    )
                except TimeoutError as e:
                    logging.error(e)
                    returncode = 1
                except Exception as e:
                    logging.error(f"{e}")
                    returncode = 1
                self.archive(tmpd_out, returncode)

    def copy_tapis_logs_to_out(self, outdirs: list[Path]) -> None:
        for rank, outdir in enumerate(outdirs):
            if rank == self.RANK and outdir.exists():
                tapis._copy_tapis_files(outdir=outdir)
            # ensure that only one copy happens at a time
            MPI.COMM_WORLD.barrier()
