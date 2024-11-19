import logging
import typing
from pathlib import Path

from biomarkers import imgs
from biomarkers.models import bids, fmriprep, signatures


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
    compcor_label: imgs.COMPCOR_LABEL | None = None,
    baseline_list: typing.Sequence[str] | None = None,
    active_list: typing.Sequence[str] | None = None,
) -> None:
    all_signatures = signatures.get_all_signatures()

    layout = bids.Layout.from_path(subdir)
    for sub in layout.subjects:
        for ses in layout.get_sessions(sub=sub):
            func3ds = signatures.gather_to_resample(
                extra=all_signatures, layout=layout, sub=sub, ses=ses, space=space
            )
            probseg = bids.ProbSeg.from_layout(
                layout=layout,
                filters={"sub": sub, "ses": ses, "space": space, "res": "2"},
            )
            flows = {}
            for task in layout.get_tasks(sub=sub, ses=ses):
                for run in layout.get_runs(sub=sub, ses=ses, task=task):
                    flow = signatures.SignatureRunFlow(
                        dst=out,
                        sub=sub,
                        ses=ses,
                        layout=layout,
                        task=task,
                        run=run,
                        space=space,
                        probseg=probseg,
                        func3ds=func3ds,
                        all_signatures=all_signatures,
                        low_pass=low_pass,
                        high_pass=high_pass,
                        n_non_steady_state_tr=n_non_steady_state_tr,
                        detrend=detrend,
                        fwhm=fwhm,
                        winsorize=winsorize,
                        compcor_label=compcor_label,
                    )
                    flow.sign_run()
                    flows[f"{task}{run}"] = flow

            if not baseline_list or not active_list:
                return

            for baseline in baseline_list:
                for active in active_list:
                    if ((rest := flows.get(baseline)) is not None) and (
                        (cuff := flows.get(active)) is not None
                    ):
                        scans = f"{active}{baseline}"
                        # may fail if baseline and active have different lengths, but we
                        # want to ignore this and continue on anyway
                        try:
                            signatures.SignatureRunPairFlow(
                                active_flow=cuff, baseline_flow=rest, scans=scans
                            ).sign_pair()
                        except AssertionError:
                            logging.error(
                                f"Failed during {scans}. Attempting to continue"
                            )
