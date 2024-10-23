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
    cwd = Path(os.environ.get("_tapisJobWorkingDir", os.getcwd()))
    for stderr in cwd.glob("*.err"):
        shutil.copyfile(stderr, outdir / stderr.name)
    for stdout in cwd.glob("*.out"):
        shutil.copyfile(stdout, outdir / stdout.name)


def _add_tapis_files_to_tarball(tarball: Path) -> None:
    # tapis logs tend to be in the form of [jobid].{err,out}
    # this copies them to a destination folder
    cwd = Path(os.environ.get("_tapisJobWorkingDir", os.getcwd()))
    with tarfile.open(tarball, "a") as tf:
        for stderr in cwd.glob("*.err"):
            tf.add(stderr, stderr.name)
        for stdout in cwd.glob("*.out"):
            tf.add(stderr, stdout.name)


class TapisEntrypoint(pydantic.BaseModel):

    ins: typing.Sequence[Path]
    outs: typing.Sequence[Path]
    stage_dir: Path

    @property
    def job_id(self) -> str:
        return os.environ.get("_tapisJobUUID", str(uuid.uuid4()))

    @abstractmethod
    def run_flow(self) -> list[Path]:
        raise NotImplementedError

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists()

    def archive(self, srcs: typing.Sequence[Path]) -> None:

        with tempfile.TemporaryDirectory() as tmpd_:
            tmpd = Path(tmpd_)
            tarballs: dict[int, Path] = {}
            try:
                for s, src in enumerate(srcs):
                    print(f"checking {src=}")
                    if self.check_outputs(src):
                        print(f"Creating tar of {src} in {tmpd}")
                        tarballs[s] = tmpd / f"uuid-{self.job_id}_rank-{s}.tar"
                        utils.recursive_chmod(src)
                        with tarfile.open(tarballs[s], mode="w") as tf:
                            tf.add(src, arcname=".")
            except Exception as e:
                print(f"Failed to tar {src}: {e}")

            # src (underneath /tmp) will automatically be deleted after the final copy but
            # archiving with tar creates a duplicate of all products, which could
            # be too much for the filesystem. So here we manually delete things.
            # But we keep the top-level files, because those will be logs
            # that might need saving
            for src in srcs:
                try:
                    print(f"Removing unarchived products {src}")
                    for item in src.glob("*"):
                        if item.is_dir():
                            shutil.rmtree(item)
                except Exception as e:
                    print(f"Failed to remove unarchived products {src}: {e}")

            for d, dst in enumerate(self.outs):
                print(f"considering {dst=}")
                try:
                    if tarball := tarballs.get(d):
                        print(f"Copying {tarball} -> {dst}")
                        if not dst.exists():
                            utils.mkdir_recursive(
                                dst, mode=utils.DIR_PERMISSIONS
                            )
                        shutil.copyfile(tarball, dst / tarball.name)
                    else:
                        # in case of failures, it's helpful to keep logs around
                        log_dst = utils.FAILURE_LOG_DST / dst.stem
                        print(
                            f"Failure detected for {dst}. Copying logs to {log_dst}"
                        )
                        if not log_dst.exists():
                            utils.mkdir_recursive(
                                log_dst, mode=utils.DIR_PERMISSIONS
                            )
                        for log in src.glob("*log"):
                            shutil.copyfile(log, log_dst / log.name)
                            _copy_tapis_files(log_dst)
                except Exception as e:
                    print(f"Failed to archive {dst=}: {e}")

    def copy_tapis_logs_to_out(self) -> None:
        print("Adding job logs to outputs")
        for outdir in self.outs:
            try:
                # when there was a success, there should be a tarball in the
                # output folder. The out,err files need to be added
                for tarball in outdir.glob("*tar"):
                    _add_tapis_files_to_tarball(tarball)
                # if there was a failure, the logs should have been copied
                # into an appropriate place underneath FAILURE_LOG_DST
                for log_dst in utils.FAILURE_LOG_DST.glob(f"*{outdir.stem}"):
                    if log_dst.is_dir():
                        _copy_tapis_files(log_dst)
            except Exception as e:
                print(f"Failed to handle job out,err: {e}")

    async def run(self) -> None:
        if not self.stage_dir.exists():
            utils.mkdir_recursive(self.stage_dir, mode=utils.DIR_PERMISSIONS)
        elif not self.stage_dir.is_dir():
            msg = f"stage_dir must be a directory. found {self.stage_dir}"
            raise AssertionError(msg)

        async with prefect_utils.get_prefect():
            staged_outs = self.run_flow()

        self.archive(staged_outs)

        # copy tapis logs at the end because archive will add more lines
        self.copy_tapis_logs_to_out()
