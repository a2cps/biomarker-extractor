import asyncio
import logging
import os
import shutil
import socket
import tarfile
import tempfile
import typing
import uuid
from abc import abstractmethod
from pathlib import Path

import pydantic
from mpi4py import MPI

from biomarkers import utils
from biomarkers.entrypoints import tapis

T = typing.TypeVar("T")


def configure_mpi_logger() -> None:
    host = socket.gethostname()
    rank = MPI.COMM_WORLD.Get_rank()
    usize = MPI.COMM_WORLD.Get_size()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | {rank=} | {usize=} | %(message)s",
        level=logging.INFO,
    )


def iterate_byrank_serial(
    items: typing.Sequence[T], RANK: int
) -> typing.Generator[T, None, None]:
    for rank, item in enumerate(items):
        if rank == RANK:
            yield item
        # ensure that only one copy happens at a time
        MPI.COMM_WORLD.barrier()


class TapisMPIEntrypoint(pydantic.BaseModel):
    ins: typing.Sequence[Path]
    outs: typing.Sequence[Path]
    timeout: int | float | None = None
    stage_ignore_patterns: (
        None | typing.Callable[[str, list[str]], typing.Iterable[str]]
    ) = None
    RANK: int = pydantic.Field(default_factory=MPI.COMM_WORLD.Get_rank)
    USIZE: int = pydantic.Field(default_factory=MPI.COMM_WORLD.Get_size)
    job_id: str = pydantic.Field(
        default_factory=lambda: os.environ.get("_tapisJobUUID", uuid.uuid4())
    )

    @abstractmethod
    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        raise NotImplementedError

    def stage(self, dst: Path) -> Path:
        for src in iterate_byrank_serial(self.ins, self.RANK):
            logging.info(f"Staging files for {src=} -> {dst=}")
            shutil.copytree(
                src,
                dst,
                ignore=self.stage_ignore_patterns,
                dirs_exist_ok=True,
            )
        return dst

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists()

    def archive(self, src: Path) -> None:
        # parallel (across mpi jobs)
        with tempfile.TemporaryDirectory() as tmpd_:
            tmpd = Path(tmpd_)
            tarballs: dict[int, Path] = {}
            try:
                if self.check_outputs(src):
                    logging.info(f"Creating tar of {src} in {tmpd}")
                    tarballs[self.RANK] = (
                        tmpd / f"uuid-{self.job_id}_rank-{self.RANK}.tar"
                    )
                    utils.recursive_chmod(src)
                    with tarfile.open(tarballs[self.RANK], mode="w") as tf:
                        tf.add(src, arcname=".")
            except Exception:
                logging.exception(f"Failed to tar {self.outs[self.RANK]=}")

            # src (underneath /tmp) will automatically be deleted after the final copy but
            # archiving with tar creates a duplicate of all products, which could
            # be too much for the filesystem. So here we manually delete things.
            # But we keep the top-level files, because those will be logs
            # that might need saving
            try:
                logging.info(f"Removing unarchived products {src}")
                for item in src.glob("*"):
                    if item.is_dir():
                        shutil.rmtree(item)
            except Exception:
                logging.exception(f"Failed to remove unarchived products {src}")

            # serial
            for dst in iterate_byrank_serial(self.outs, self.RANK):
                try:
                    if tarball := tarballs.get(self.RANK):
                        logging.info(f"Copying {tarball} -> {dst}")
                        if not dst.exists():
                            utils.mkdir_recursive(dst, mode=utils.DIR_PERMISSIONS)
                        shutil.copyfile(tarball, dst / tarball.name)
                    else:
                        # in case of failures, it's helpful to keep logs around
                        log_dst = utils.FAILURE_LOG_DST / dst.stem
                        logging.warning(
                            f"Failure detected for {self.outs[self.RANK]=}. Copying logs to {log_dst}"
                        )
                        if not log_dst.exists():
                            utils.mkdir_recursive(log_dst, mode=utils.DIR_PERMISSIONS)
                        for log in src.glob("*log"):
                            shutil.copyfile(log, log_dst / log.name)
                            tapis._copy_tapis_files(log_dst)
                except Exception:
                    logging.exception(f"Failed to archive {self.outs[self.RANK]=}")

    async def run(self):
        with tempfile.TemporaryDirectory() as _tmpd_in:
            tmpd_in = self.stage(dst=Path(_tmpd_in))
            with tempfile.TemporaryDirectory() as _tmpd_out:
                tmpd_out = Path(_tmpd_out)
                try:
                    # jobs could get stuck for one process, which would prevent other tasks
                    # from archiving outputs. This forces an error if things have been
                    # going for too long
                    await asyncio.wait_for(
                        self.run_flow(tmpd_in, tmpd_out), timeout=self.timeout
                    )
                except Exception:
                    logging.exception("Flow failed")
                finally:
                    self.archive(tmpd_out)

                    # copy tapis logs at the end because archive will add more lines
                    self.copy_tapis_logs_to_out()

    def copy_tapis_logs_to_out(self) -> None:
        logging.info("Adding job logs to outputs")
        for outdir in iterate_byrank_serial(self.outs, self.RANK):
            try:
                # when there was a success, there should be a tarball in the
                # output folder. The out,err files need to be added
                for tarball in outdir.glob("*tar"):
                    tapis._add_tapis_files_to_tarball(tarball)
                # if there was a failure, the logs should have been copied
                # into an appropriate place underneath FAILURE_LOG_DST
                for log_dst in utils.FAILURE_LOG_DST.glob(f"*{outdir.stem}"):
                    if log_dst.is_dir():
                        tapis._copy_tapis_files(log_dst)
            except Exception:
                logging.exception("Failed to handle job out,err")
