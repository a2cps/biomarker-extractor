import functools
from pathlib import Path

import polars as pl
import pydantic

from biomarkers import imgs, utils
from biomarkers.models import bids, fmriprep


class PostProcessRunFlow(pydantic.BaseModel):
    dst: Path
    sub: str
    ses: str
    layout: bids.Layout
    task: str
    run: str
    space: fmriprep.SPACE
    low_pass: float | None = None
    high_pass: float | None = None
    n_non_steady_state_tr: int = 0
    detrend: bool = False
    fwhm: float | None = None
    winsorize: bool = True
    res: str = "2"
    compcor_label: imgs.COMPCOR_LABEL | None = None

    @property
    def filter(self) -> dict[str, str]:
        return {"sub": self.sub, "ses": self.ses, "task": self.task, "run": self.run}

    @property
    def filter_with_space(self) -> dict[str, str]:
        return {**self.filter, "space": self.space, "resolution": self.res}

    @property
    def preproc(self) -> Path:
        return self.layout.get_preproc(filters=self.filter_with_space)

    @property
    def boldref(self) -> Path:
        return self.layout.get_boldref(filters=self.filter_with_space)

    @property
    def mask(self) -> Path:
        return self.layout.get_brain(filters=self.filter_with_space)

    @property
    def cleaned(self) -> Path:
        return self.dst / "cleaned" / self.preproc.name

    @property
    def confounds(self) -> Path:
        return (
            self.dst
            / "confounds"
            / f"sub={self.sub}"
            / f"ses={self.ses}"
            / f"task={self.task}"
            / f"run={self.run}"
            / "part-0.parquet"
        )

    @functools.cached_property
    def probseg(self) -> bids.ProbSeg:
        return bids.ProbSeg.from_layout(
            self.layout,
            filters={"sub": self.sub, "ses": self.ses, "space": self.space, "res": "2"},
        )

    def clean(self) -> pl.DataFrame:
        if self.compcor_label:
            compcor = imgs.CompCor(
                img=self.preproc,
                probseg=self.probseg,
                label=self.compcor_label,
                boldref=self.boldref,
                high_pass=self.high_pass,
                low_pass=self.low_pass,
                n_non_steady_state_tr=self.n_non_steady_state_tr,
                detrend=self.detrend,
            ).fit_transform()
        else:
            compcor = None

        utils.update_confounds(
            self.confounds,
            confounds=self.layout.get_confounds(filters=self.filter),
            n_non_steady_state_tr=self.n_non_steady_state_tr,
            compcor=compcor,
        )

        imgs.clean_img(
            self.cleaned,
            img=self.preproc,
            mask=self.mask,
            confounds_file=self.confounds,
            high_pass=self.high_pass,
            low_pass=self.low_pass,
            do_detrend=self.detrend,
            fwhm=self.fwhm,
            do_winsorize=self.winsorize,
            to_percentchange=False,
            n_non_steady_state_tr=self.n_non_steady_state_tr,
        )
        return bids.Func4d(path=self.cleaned).to_polars()
