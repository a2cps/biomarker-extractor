from pathlib import Path

from biomarkers import imgs
from biomarkers.models import bids, functional_connectivity, postprocess


def connectivity_flow(
    subdir: Path,
    out: Path,
    compcor_label: imgs.COMPCOR_LABEL | None = None,
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 15,
    detrend: bool = False,
    fwhm: float | None = None,
    winsorize: bool = True,
) -> None:
    layout = bids.Layout.from_path(subdir)

    for sub in layout.subjects:
        for ses in layout.get_sessions(sub=sub):
            for task in layout.get_tasks(sub=sub, ses=ses):
                for run in layout.get_runs(sub=sub, ses=ses, task=task):
                    process_flow = postprocess.PostProcessRunFlow(
                        dst=out,
                        sub=sub,
                        ses=ses,
                        layout=layout,
                        task=task,
                        run=run,
                        space="MNI152NLin2009cAsym",
                        low_pass=low_pass,
                        high_pass=high_pass,
                        n_non_steady_state_tr=n_non_steady_state_tr,
                        detrend=detrend,
                        fwhm=fwhm,
                        winsorize=winsorize,
                        compcor_label=compcor_label,
                    )
                    functional_connectivity.PostProcessRunFlow(
                        process_flow=process_flow, coordinates=None, labels=None
                    ).run()
                    process_flow = postprocess.PostProcessRunFlow(
                        dst=out,
                        sub=sub,
                        ses=ses,
                        layout=layout,
                        task=task,
                        run=run,
                        space="MNI152NLin6Asym",
                        low_pass=low_pass,
                        high_pass=high_pass,
                        n_non_steady_state_tr=n_non_steady_state_tr,
                        detrend=detrend,
                        fwhm=fwhm,
                        winsorize=winsorize,
                        compcor_label=compcor_label,
                    )
                    functional_connectivity.PostProcessRunFlow(
                        process_flow=process_flow, maps=None
                    ).run()
