from pathlib import Path

import ancpbids
import polars as pl

from biomarkers import datasets, imgs, utils
from biomarkers.models import fmriprep, signatures


def cosine_similarity(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).dot(pl.col(col2)) / (
        pl.col(col1).pow(2).sum().sqrt() * pl.col(col2).pow(2).sum().sqrt()
    )


def gather_to_resample(
    extra: list[signatures.Func3d],
    layout: ancpbids.BIDSLayout,
    sub: str,
    ses: str,
    space: fmriprep.SPACE = "MNI152NLin6Asym",
) -> list[signatures.Func3d]:
    filters = {"sub": sub, "ses": ses, "space": space, "extension": ".nii.gz"}
    GM = layout.get(label="GM", return_type="filename", **filters)[0]
    WM = layout.get(label="WM", return_type="filename", **filters)[0]
    CSF = layout.get(label="CSF", return_type="filename", **filters)[0]
    return [
        signatures.Func3d(label="GM", dtype="f8", path=Path(str(GM))),
        signatures.Func3d(label="CSF", dtype="f8", path=Path(str(CSF))),
        signatures.Func3d(label="WM", dtype="f8", path=Path(str(WM))),
    ] + extra


def get_in_space(
    layout: ancpbids.BIDSLayout,
    sub: str,
    ses: str,
    task: str,
    run: str,
    space: fmriprep.SPACE = "MNI152NLin6Asym",
) -> list[signatures.Func3d]:
    filters = {
        "sub": sub,
        "ses": ses,
        "space": space,
        "task": task,
        "run": run,
        "extension": ".nii.gz",
    }
    masks = layout.get(desc="brain", return_type="filename", **filters)
    if not len(masks) == 1:
        msg = f"expected 1 mask with {filters} but found {masks}"
        raise RuntimeError(msg)
    return [signatures.Func3d(label="brain", dtype="?", path=Path(str(masks[0])))]


def get(layout: ancpbids.BIDSLayout, filters: dict[str, str]) -> Path:
    file = layout.get(return_type="filename", **filters)
    if not len(file) == 1:
        msg = (
            f"Expected that only 1 file would be retreived but saw {file=}; {filters=}"
        )
        raise ValueError(msg)
    return Path(str(file[0]))


def get_all_signatures() -> list[signatures.Func3d]:
    return [
        signatures.Func3d(path=datasets.get_nps(x), label=x, dtype="f8")
        for x in signatures.NPS
    ] + [
        signatures.Func3d(path=datasets.get_siips1(x), label=x, dtype="f8")
        for x in signatures.SIIPS
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


def signature_flow(
    subdir: Path,
    out: Path,
    high_pass: float | None = None,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    detrend: bool = True,
    fwhm: float | None = None,
    winsorize: bool = True,
    space: fmriprep.SPACE = "MNI152NLin6Asym",
) -> None:
    all_signatures = get_all_signatures()

    layout = ancpbids.BIDSLayout(str(subdir))
    for sub in layout.get_subjects():
        for ses in layout.get_sessions(sub=sub):
            func3ds = gather_to_resample(
                extra=all_signatures,
                layout=layout,
                sub=str(sub),
                ses=str(ses),
                space=space,
            )

            for task in layout.get_tasks(sub=sub, ses=ses):
                for run in layout.get_runs(sub=sub, ses=ses, task=task):
                    confounds = utils.update_confounds(
                        out
                        / "signature-confounds"
                        / f"sub={sub}"
                        / f"ses={ses}"
                        / f"task={task}"
                        / f"run={run}"
                        / "part-0.parquet",
                        confounds=get(
                            layout=layout,
                            filters={
                                "sub": str(sub),
                                "ses": str(ses),
                                "task": str(task),
                                "run": str(run),
                                "desc": "confounds",
                                "extension": ".tsv",
                            },
                        ),
                        n_non_steady_state_tr=n_non_steady_state_tr,
                    )
                    preproc = get(
                        layout=layout,
                        filters={
                            "sub": str(sub),
                            "ses": str(ses),
                            "task": str(task),
                            "run": str(run),
                            "space": space,
                            "desc": "preproc",
                            "suffix": "bold",
                            "extension": ".nii.gz",
                        },
                    )
                    mask = get(
                        layout=layout,
                        filters={
                            "sub": str(sub),
                            "ses": str(ses),
                            "task": str(task),
                            "run": str(run),
                            "space": space,
                            "desc": "brain",
                            "suffix": "mask",
                            "extension": ".nii.gz",
                        },
                    )

                    cleaned = imgs.clean_img(
                        out / "signature-cleaned" / preproc.name,
                        img=preproc,
                        mask=mask,
                        confounds_file=confounds,
                        high_pass=high_pass,
                        low_pass=low_pass,
                        do_detrend=detrend,
                        fwhm=fwhm,
                        do_winsorize=winsorize,
                        to_percentchange=False,
                        n_non_steady_state_tr=n_non_steady_state_tr,
                    )
                    bold = signatures.Func4d(path=cleaned).to_polars()

                    fmriprep_func3ds = get_in_space(
                        layout=layout,
                        sub=str(sub),
                        ses=str(ses),
                        task=str(task),
                        run=str(run),
                        space=space,
                    )
                    labels = utils.to_parquet3d(
                        out
                        / "signature-labels"
                        / f"sub={sub}"
                        / f"ses={ses}"
                        / f"task={task}"
                        / f"run={run}"
                        / "part-0.parquet",
                        func3ds=func3ds,
                        fmriprep_func3ds=fmriprep_func3ds,
                    )

                    sign_by_run(
                        out
                        / "signature-by-run"
                        / f"sub={sub}"
                        / f"ses={ses}"
                        / f"task={task}"
                        / f"run={run}"
                        / "part-0.parquet",
                        bold=bold,
                        labels=labels,
                        signatures=[x.label for x in all_signatures],
                    )
                    sign_by_part(
                        out
                        / "signature-by-part"
                        / f"sub={sub}"
                        / f"ses={ses}"
                        / f"task={task}"
                        / f"run={run}"
                        / "part-0.parquet",
                        bold=bold,
                        labels=labels,
                        signatures=[x.label for x in all_signatures],
                    )
                    sign_by_t(
                        out
                        / "signature-by-tr"
                        / f"sub={sub}"
                        / f"ses={ses}"
                        / f"task={task}"
                        / f"run={run}"
                        / "part-0.parquet",
                        bold=bold,
                        labels=labels,
                        signatures=[x.label for x in all_signatures],
                    )
