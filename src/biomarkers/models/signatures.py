import functools
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
import pydantic
from nilearn import image

from biomarkers import datasets, imgs, utils
from biomarkers.models import bids, fmriprep

NPS: tuple[datasets.NPSWeights, ...] = (
    "grouppred_cvpcr_FDR05_smoothed_fwhm05",
    "grouppred_cvpcr",
    "positive_smoothed_larger_than_10vox",
    "positive_smoothed_larger_than_10vox_roi-dACC",
    "positive_smoothed_larger_than_10vox_roi-lIns",
    "positive_smoothed_larger_than_10vox_roi-rdpIns",
    "positive_smoothed_larger_than_10vox_roi-rIns",
    "positive_smoothed_larger_than_10vox_roi-rS2_Op",
    "positive_smoothed_larger_than_10vox_roi-rThal",
    "positive_smoothed_larger_than_10vox_roi-rV1",
    "positive_smoothed_larger_than_10vox_roi-vermis",
    "negative_smoothed_larger_than_10vox",
    "negative_smoothed_larger_than_10vox_roi-lLOC",
    "negative_smoothed_larger_than_10vox_roi-lSTS",
    "negative_smoothed_larger_than_10vox_roi-PCC",
    "negative_smoothed_larger_than_10vox_roi-pgACC",
    "negative_smoothed_larger_than_10vox_roi-rIPL",
    "negative_smoothed_larger_than_10vox_roi-rLOC",
    "negative_smoothed_larger_than_10vox_roi-rpLOC",
)

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
    func3ds: typing.Sequence[bids.Func3d]
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
        return {**self.filter, "space": self.space, "res": self.res}

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

    @property
    def signatures(self) -> list[str]:
        return [x.label for x in self.all_signatures]

    @property
    def labels(self) -> Path:
        return (
            self.dst
            / "signatures-labels"
            / f"sub={self.sub}"
            / f"ses={self.ses}"
            / f"task={self.task}"
            / f"run={self.run}"
            / "part-0.parquet"
        )

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

        fmriprep_func3ds = get_in_space(
            layout=self.layout,
            sub=self.sub,
            ses=self.ses,
            task=self.task,
            run=self.run,
            space=self.space,
        )
        utils.to_parquet3d(
            self.labels, func3ds=self.func3ds, fmriprep_func3ds=fmriprep_func3ds
        )

        sign_by_run(
            self.by_run, bold=bold, labels=self.labels, signatures=self.signatures
        )
        sign_by_part(
            self.by_part, bold=bold, labels=self.labels, signatures=self.signatures
        )
        sign_by_t(self.by_tr, bold=bold, labels=self.labels, signatures=self.signatures)


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
        active = nb.nifti1.Nifti1Image.load(self.active_flow.cleaned)
        baseline = nb.nifti1.Nifti1Image.load(self.baseline_flow.cleaned)
        if np.allclose(active.shape, baseline.shape):
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
    def labels(self) -> Path:
        return self.active_flow.labels

    @property
    def signatures(self) -> list[str]:
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
        sign_by_run(
            self.by_run, bold=self.bold, labels=self.labels, signatures=self.signatures
        )
        sign_by_part(
            self.by_part, bold=self.bold, labels=self.labels, signatures=self.signatures
        )
        sign_by_t(
            self.by_tr, bold=self.bold, labels=self.labels, signatures=self.signatures
        )


def cosine_similarity(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).dot(pl.col(col2)) / (
        pl.col(col1).pow(2).sum().sqrt() * pl.col(col2).pow(2).sum().sqrt()
    )


def gather_to_resample(
    extra: list[bids.Func3d],
    layout: bids.Layout,
    sub: str,
    ses: str,
    space: fmriprep.SPACE = "MNI152NLin6Asym",
) -> list[bids.Func3d]:
    filters = {"sub": sub, "ses": ses, "space": space, "res": "2"}
    GM = layout.get_gm(filters)
    WM = layout.get_wm(filters)
    CSF = layout.get_csf(filters)
    return [
        bids.Func3d(label="GM", dtype="f8", path=Path(str(GM))),
        bids.Func3d(label="CSF", dtype="f8", path=Path(str(CSF))),
        bids.Func3d(label="WM", dtype="f8", path=Path(str(WM))),
    ] + extra


def get_in_space(
    layout: bids.Layout,
    sub: str,
    ses: str,
    task: str,
    run: str,
    space: fmriprep.SPACE = "MNI152NLin6Asym",
) -> list[bids.Func3d]:
    filters = {
        "sub": sub,
        "ses": ses,
        "space": space,
        "task": task,
        "run": run,
        "extension": ".nii.gz",
    }
    masks = layout.get_brain(filters)
    return [bids.Func3d(label="brain", dtype="?", path=masks)]


def get_all_signatures() -> list[bids.Func3d]:
    return [bids.Func3d(path=datasets.get_nps(x), label=x, dtype="f8") for x in NPS] + [
        bids.Func3d(path=datasets.get_siips1(x), label=x, dtype="f8") for x in SIIPS
    ]


@utils.cache_dataframe
def sign_by_t(bold: pl.DataFrame, labels: Path, signatures: list[str]) -> pl.DataFrame:
    # for memory reasons, iterate and aggregate
    la = (
        pl.scan_parquet(labels)
        .filter(pl.col("brain"))
        .select(["voxel", *signatures])
        .collect()
    )
    out = []
    for _, bo in bold.group_by("t"):
        out.append(
            bo.join(la, on="voxel", how="inner", validate="m:1")
            .unpivot(index=["voxel", "t", "signal"], variable_name="signature")
            .group_by("t", "signature")
            .agg(
                correlation=pl.corr(pl.col("value"), pl.col("signal")),
                dot=pl.col("signal").dot(pl.col("value")),
                cosine=cosine_similarity("signal", "value"),
            )
        )

    return pl.concat(out)


@utils.cache_dataframe
def sign_by_run(
    bold: pl.DataFrame, labels: Path, signatures: list[str]
) -> pl.DataFrame:
    bo = bold.group_by("voxel").agg(signal=pl.col("signal").mean()).lazy()
    la = pl.scan_parquet(labels).filter(pl.col("brain")).select(["voxel", *signatures])
    return (
        bo.join(la, on="voxel", how="inner")
        .unpivot(index=["voxel", "signal"], variable_name="signature")
        .group_by("signature")
        .agg(
            correlation=pl.corr(pl.col("value"), pl.col("signal")),
            dot=pl.col("signal").dot(pl.col("value")),
            cosine=cosine_similarity("signal", "value"),
        )
        .collect()
    )


@utils.cache_dataframe
def sign_by_part(
    bold: pl.DataFrame,
    labels: Path,
    signatures: list[str],
    bins: tuple[float, ...] = (138.0, 300.0),
    bin_labels: tuple[str, ...] = ("beginning", "middle", "end"),
) -> pl.DataFrame:
    bo = (
        bold.with_columns(
            pl.col("t").cut(breaks=list(bins), labels=list(bin_labels)).alias("part")
        )
        .group_by("voxel", "part")
        .agg(signal=pl.col("signal").mean())
        .lazy()
    )
    la = pl.scan_parquet(labels).filter(pl.col("brain")).select(["voxel", *signatures])
    return (
        bo.join(la, on="voxel", how="inner", validate="m:m")
        .unpivot(index=["voxel", "part", "signal"], variable_name="signature")
        .group_by("signature", "part")
        .agg(
            correlation=pl.corr(pl.col("value"), pl.col("signal")),
            dot=pl.col("signal").dot(pl.col("value")),
            cosine=cosine_similarity("signal", "value"),
        )
        .collect()
    )
