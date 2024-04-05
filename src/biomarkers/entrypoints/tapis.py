import logging
import shutil
import typing
from abc import abstractmethod
from pathlib import Path

import pydantic

from biomarkers import prefect_utils


class TapisEntrypoint(pydantic.BaseModel):

    ins: typing.Iterable[pydantic.DirectoryPath]
    outs: typing.Iterable[Path]
    stage_dir: Path
    n_workers: int = 1

    @abstractmethod
    def run_flow(self) -> list[Path]:
        raise NotImplementedError

    def check_outputs(self, *args, **kwargs) -> bool:
        return True

    def run(self) -> None:
        if not self.stage_dir.exists():
            self.stage_dir.mkdir(parents=True)
        elif not self.stage_dir.is_dir():
            msg = f"stage_dir must be a directory. found {self.stage_dir}"
            raise AssertionError(msg)

        with prefect_utils.get_prefect():
            staged_outs = self.run_flow()

        for o in staged_outs:
            if o.exists() and self.check_outputs(o):

                # tapis log files to dsts
                for stderr in Path.cwd().glob("*.err"):
                    shutil.copy2(stderr, o)
                for stdout in Path.cwd().glob("*.out"):
                    shutil.copy2(stdout, o)

                # dirs_exist_ok=True because we are likely gathering results
                # from multiple sources (e.g., a run of fmriprep-cuff and fmriprep-rest)
                # will end up in the same folder
                shutil.copytree(
                    o, o.relative_to(self.stage_dir), dirs_exist_ok=True
                )
            else:
                logging.warning(f"Expected {o} but that path does not exist")

        if self.stage_dir:
            shutil.rmtree(self.stage_dir)
