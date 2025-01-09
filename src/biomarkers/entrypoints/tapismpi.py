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

T = typing.TypeVar("T")


def copy_tapis_files(outdir: Path) -> None:
    # tapis logs tend to be in the form of [jobid].{err,out}
    # this copies them to a destination folder
    cwd = Path(os.environ.get("_tapisJobWorkingDir", os.getcwd()))
    for stderr in cwd.glob("*.err"):
        shutil.copyfile(stderr, outdir / stderr.name)
    for stdout in cwd.glob("*.out"):
        shutil.copyfile(stdout, outdir / stdout.name)


def add_tapis_files_to_tarball(tarball: Path) -> None:
    # tapis logs tend to be in the form of [jobid].{err,out}
    # this copies them to a destination folder
    cwd = Path(os.environ.get("_tapisJobWorkingDir", os.getcwd()))
    with tarfile.open(tarball, "a") as tf:
        for stderr in cwd.glob("*.err"):
            tf.add(stderr, stderr.name)
        for stdout in cwd.glob("*.out"):
            tf.add(stdout, stdout.name)


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
        default_factory=lambda: os.environ.get("_tapisJobUUID", str(uuid.uuid4()))
    )

    @abstractmethod
    async def run_flow(self, in_dir: Path, out_dir: Path) -> None:
        raise NotImplementedError

    def stage(self, dst: Path) -> Path:
        for src in iterate_byrank_serial(self.ins, self.RANK):
            logging.info(f"Staging files for {src=} -> {dst=}")
            try:
                shutil.copytree(
                    src, dst, ignore=self.stage_ignore_patterns, dirs_exist_ok=True
                )
            except Exception:
                logging.error("Failed to stage. Subsequent steps will likely fail.")
        return dst

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists()

    def archive(self, src: Path) -> None:
        # parallel (across mpi jobs)
        tarballs: dict[int, Path] = {}
        try:
            if self.check_outputs(src):
                tarballs[self.RANK] = (
                    self.outs[self.RANK] / f"uuid-{self.job_id}_rank-{self.RANK}.tar"
                )
                utils.recursive_chmod(src)
            else:
                logging.warning(f"Failure detected for {self.outs[self.RANK]=}")
        except Exception:
            logging.exception(f"Failed to chmod {self.outs[self.RANK]=}")
            if tarballs.get(self.RANK):
                del tarballs[self.RANK]

        # serial
        for dst in iterate_byrank_serial(self.outs, self.RANK):
            try:
                if tarball := tarballs.get(self.RANK):
                    logging.info(f"Making {tarball}")
                    if not dst.exists():
                        utils.mkdir_recursive(dst, mode=utils.DIR_PERMISSIONS)
                    with tarfile.open(tarball, mode="w") as tf:
                        tf.add(src, arcname=".")
                    logging.info("Finished tar")
            except Exception:
                logging.exception(f"Failed to archive {self.outs[self.RANK]=}")
                if tf := tarballs.get(self.RANK):
                    if tf.exists():
                        tf.unlink()
                    # delete so that failures to tar still trigger copying of
                    # failure logs
                    del tarballs[self.RANK]

            if tarballs.get(self.RANK) is None:
                # in case of failures, it's helpful to keep logs around
                log_dst = utils.FAILURE_LOG_DST / dst.stem
                logging.warning(
                    f"Failure detected for {self.outs[self.RANK]=}. Copying logs to {log_dst}"
                )
                if not log_dst.exists():
                    utils.mkdir_recursive(log_dst, mode=utils.DIR_PERMISSIONS)
                for log in src.glob("*log"):
                    shutil.copyfile(log, log_dst / log.name)
                    copy_tapis_files(log_dst)

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
        for outdir in iterate_byrank_serial(self.outs, self.RANK):
            logging.info("Adding job logs to outputs")
            try:
                # when there was a success, there should be a tarball in the
                # output folder. The out,err files need to be added
                for tarball in outdir.glob("*tar"):
                    add_tapis_files_to_tarball(tarball)
                # if there was a failure, the logs should have been copied
                # into an appropriate place underneath FAILURE_LOG_DST
                for log_dst in utils.FAILURE_LOG_DST.glob(f"*{outdir.stem}"):
                    if log_dst.is_dir():
                        copy_tapis_files(log_dst)
            except Exception:
                logging.exception("Failed to handle job out,err")
