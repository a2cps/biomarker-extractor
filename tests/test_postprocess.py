import typing
from pathlib import Path

import pytest

from biomarkers import imgs
from biomarkers.models import bids, postprocess


@pytest.mark.parametrize("high_pass", [0.01, None])
@pytest.mark.parametrize("low_pass", [0.1, None])
@pytest.mark.parametrize("detrend", [True, False])
@pytest.mark.parametrize("winsorize", [True, False])
@pytest.mark.parametrize("compcor_label", typing.get_args(imgs.COMPCOR_LABEL))
def test_sig_flow(
    tmp_path: Path,
    high_pass: float | None,
    low_pass: float | None,
    detrend: bool,
    winsorize: bool,
    compcor_label: imgs.COMPCOR_LABEL,
):
    sub = "travel2"
    ses = "RU"
    task = "rest"
    run = "01"
    space = "MNI152NLin6Asym"
    layout = bids.Layout.from_path(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep")
    )
    postprocess.PostProcessRunFlow(
        dst=tmp_path,
        sub=sub,
        ses=ses,
        layout=layout,
        task=task,
        run=run,
        space=space,
        high_pass=high_pass,
        low_pass=low_pass,
        n_non_steady_state_tr=350,  # for speed
        detrend=detrend,
        winsorize=winsorize,
        compcor_label=compcor_label,
    ).clean()

    exists = [(tmp_path / d).exists() for d in ("cleaned", "confounds")]
    assert all(exists)
