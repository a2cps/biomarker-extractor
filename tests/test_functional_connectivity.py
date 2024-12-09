from pathlib import Path

from sklearn import covariance

from biomarkers.flows import functional_connectivity as fnc_flow
from biomarkers.models import bids, functional_connectivity, postprocess


def test_fnc_model_quick(tmp_path: Path):
    sub = "travel2"
    ses = "RU"
    layout = bids.Layout.from_path(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep")
    )
    process_flow = postprocess.PostProcessRunFlow(
        dst=tmp_path,
        sub=sub,
        ses=ses,
        layout=layout,
        task="rest",
        run="01",
        space="MNI152NLin6Asym",
        n_non_steady_state_tr=400,
        detrend=False,
        winsorize=False,
    )
    functional_connectivity.PostProcessRunFlow(
        process_flow=process_flow,
        estimators={"empirical": covariance.EmpiricalCovariance},
        coordinates={"dmn": functional_connectivity.get_baliki_coordinates()},
    ).run()

    exists = [
        (tmp_path / d).exists()
        for d in ("connectivity", "cleaned", "confounds", "timeseries")
    ]
    assert all(exists)


def test_fnc_flow(tmp_path: Path):
    fnc_flow.connectivity_flow(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep"),
        out=tmp_path,
        space="MNI152NLin6Asym",
        n_non_steady_state_tr=15,
    )

    exists = [
        (tmp_path / d).exists()
        for d in ("connectivity", "cleaned", "confounds", "timeseries")
    ]
    assert all(exists)
