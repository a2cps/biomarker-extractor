from pathlib import Path

import prefect

import ancpbids

from biomarkers.flows import signature


@prefect.task()
def _gather_anat(
    layout: ancpbids.BIDSLayout,
    sub: str,
    ses: str,
    space: str = "MNI152NLin2009cAsym",
) -> list[signature.Func3d]:
    filters = {"sub": sub, "ses": ses, "space": space}
    out = []
    for label in ["GM", "WM", "CSF"]:
        out.append(
            signature.Func3d(
                label=label,
                dtype="f8",
                path=Path(
                    str(layout.get(label=label, return_type="filename", **filters)[0])
                ),
            )
        )
    for desc in ["brain", "preproc"]:
        out.append(
            signature.Func3d(
                label=desc,
                dtype="f8",
                path=Path(
                    str(layout.get(desc=desc, return_type="filename", **filters)[0])
                ),
            )
        )

    return out


@prefect.flow()
def anat_flow(
    subdirs: frozenset[Path], out: Path, space: str = "MNI152NLin2009cAsym"
) -> None:
    for subdir in subdirs:
        layout = ancpbids.BIDSLayout(str(subdir))
        for sub in layout.get_subjects():
            for ses in layout.get_sessions(sub=sub):
                to_convert = _gather_anat.submit(
                    layout=layout, sub=str(sub), ses=str(ses), space=space
                )

                signature.to_parquet3d.submit(
                    out / "anat" / f"sub={sub}" / f"ses={ses}" / "part-0.parquet",
                    fmriprep_func3ds=to_convert,  # type: ignore
                )
