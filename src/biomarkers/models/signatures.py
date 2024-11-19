import functools
import typing
from pathlib import Path

import nibabel as nb
import polars as pl
import pydantic
from nilearn import image

from biomarkers import datasets, imgs, utils
from biomarkers.models import bids, fmriprep

NPS: tuple[datasets.NPSWeights, ...] = typing.get_args(datasets.NPSWeights)

SIIPS: tuple[datasets.SIIPS1Weights, ...] = ("137subjmap_weighted_mean",)


class SignatureRunFlow(pydantic.BaseModel):
    dst: Path
    sub: str
    ses: str
    layout: bids.Layout
    task: str
    run: str
    space: fmriprep.SPACE
    probseg: bids.ProbSeg
    all_signatures: typing.Sequence[bids.Func3d]
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
        return self.dst / "signatures-cleaned" / self.preproc.name

    @functools.cached_property
    def signatures(self) -> pl.DataFrame:
        target = nb.nifti1.Nifti1Image.load(self.boldref)
        return pl.concat(
            [
                s.to_polars(target=target)
                .rename({s.label: "value"})
                .with_columns(signature=pl.lit(s.label))
                for s in self.all_signatures
            ]
        ).filter(pl.col("value").abs() > 0)

    @property
    def confounds(self) -> Path:
        return (
            self.dst
            / "signatures-confounds"
            / f"sub={self.sub}"
            / f"ses={self.ses}"
            / f"task={self.task}"
            / f"run={self.run}"
            / "part-0.parquet"
        )

    def get_by_path(self, root: str) -> Path:
        return (
            self.dst
            / root
            / f"sub={self.sub}"
            / f"ses={self.ses}"
            / f"task={self.task}"
            / f"run={self.run}"
            / "part-0.parquet"
        )

    @property
    def by_run(self) -> Path:
        return self.get_by_path("signatures-by-run")

    @property
    def by_part(self) -> Path:
        return self.get_by_path("signatures-by-part")

    @property
    def by_tr(self) -> Path:
        return self.get_by_path("signatures-by-tr")

    def sign_run(self):
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
        bold = bids.Func4d(path=self.cleaned).to_polars()

        sign_by_run(self.by_run, bold=bold, signatures=self.signatures)
        sign_by_part(self.by_part, bold=bold, signatures=self.signatures)
        sign_by_t(self.by_tr, bold=bold, signatures=self.signatures)


class SignatureRunPairFlow(pydantic.BaseModel):
    active_flow: SignatureRunFlow
    baseline_flow: SignatureRunFlow
    scans: str

    @pydantic.model_validator(mode="after")
    def check_dsts_match(self) -> typing.Self:
        if self.active_flow.dst != self.baseline_flow.dst:
            raise AssertionError("dsts do not match!")
        return self

    @pydantic.model_validator(mode="after")
    def check_subs_match(self) -> typing.Self:
        if self.active_flow.sub != self.baseline_flow.sub:
            raise AssertionError("subs do not match!")
        return self

    @pydantic.model_validator(mode="after")
    def check_ses_match(self) -> typing.Self:
        if self.active_flow.ses != self.baseline_flow.ses:
            raise AssertionError("ses do not match!")
        return self

    @pydantic.model_validator(mode="after")
    def check_cleaned_match(self) -> typing.Self:
        if not utils.check_matching_image_shapes(
            [self.active_flow.cleaned, self.baseline_flow.cleaned]
        ):
            raise AssertionError("cleaned shapes do not match!")
        return self

    @property
    def dst(self) -> Path:
        return self.active_flow.dst

    @property
    def sub(self) -> str:
        return self.active_flow.sub

    @property
    def ses(self) -> str:
        return self.active_flow.ses

    @property
    def signatures(self) -> pl.DataFrame:
        return self.active_flow.signatures

    @functools.cached_property
    def bold(self) -> pl.DataFrame:
        # assuming files exist at this point
        bold_nii: nb.nifti1.Nifti1Image = image.math_img(
            "img1 - img2",
            img1=self.active_flow.cleaned,
            img2=self.baseline_flow.cleaned,
        )  # type: ignore
        return bids.from_4d_to_polars(bold_nii)

    def get_by_path(self, root: str) -> Path:
        return (
            self.dst
            / root
            / f"sub={self.sub}"
            / f"ses={self.ses}"
            / f"scans={self.scans}"
            / "part-0.parquet"
        )

    @property
    def by_run(self) -> Path:
        return self.get_by_path("signatures-by-run-diff")

    @property
    def by_part(self) -> Path:
        return self.get_by_path("signatures-by-part-diff")

    @property
    def by_tr(self) -> Path:
        return self.get_by_path("signatures-by-tr-diff")

    def sign_pair(self):
        sign_by_run(self.by_run, bold=self.bold, signatures=self.signatures)
        sign_by_part(self.by_part, bold=self.bold, signatures=self.signatures)
        sign_by_t(self.by_tr, bold=self.bold, signatures=self.signatures)


def cosine_similarity(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).dot(pl.col(col2)) / (
        pl.col(col1).pow(2).sum().sqrt() * pl.col(col2).pow(2).sum().sqrt()
    )


def get_all_signatures() -> list[bids.Func3d]:
    return [bids.Func3d(path=datasets.get_nps(x), label=x, dtype="f8") for x in NPS] + [
        bids.Func3d(path=datasets.get_siips1(x), label=x, dtype="f8") for x in SIIPS
    ]


@utils.cache_dataframe
def sign_by_t(bold: pl.DataFrame, signatures: pl.DataFrame) -> pl.DataFrame:
    # for memory reasons, iterate and aggregate
    out = []
    for _, bo in bold.group_by("t"):
        out.append(
            bo.join(signatures, on="voxel", how="inner", validate="m:m")
            .group_by("t", "signature")
            .agg(
                correlation=pl.corr(pl.col("value"), pl.col("signal")),
                dot=pl.col("signal").dot(pl.col("value")),
                cosine=cosine_similarity("signal", "value"),
            )
        )

    return pl.concat(out)


@utils.cache_dataframe
def sign_by_run(bold: pl.DataFrame, signatures: pl.DataFrame) -> pl.DataFrame:
    bo = bold.group_by("voxel").agg(signal=pl.col("signal").mean())
    return (
        bo.join(signatures, on="voxel", how="inner")
        .group_by("signature")
        .agg(
            correlation=pl.corr(pl.col("value"), pl.col("signal")),
            dot=pl.col("signal").dot(pl.col("value")),
            cosine=cosine_similarity("signal", "value"),
        )
    )


@utils.cache_dataframe
def sign_by_part(
    bold: pl.DataFrame,
    signatures: pl.DataFrame,
    bins: tuple[float, ...] = (138.0, 300.0),
    bin_labels: tuple[str, ...] = ("beginning", "middle", "end"),
) -> pl.DataFrame:
    bo = (
        bold.with_columns(
            pl.col("t").cut(breaks=list(bins), labels=list(bin_labels)).alias("part")
        )
        .group_by("voxel", "part")
        .agg(signal=pl.col("signal").mean())
    )
    return (
        bo.join(signatures, on="voxel", how="inner", validate="m:m")
        .group_by("signature", "part")
        .agg(
            correlation=pl.corr(pl.col("value"), pl.col("signal")),
            dot=pl.col("signal").dot(pl.col("value")),
            cosine=cosine_similarity("signal", "value"),
        )
    )
