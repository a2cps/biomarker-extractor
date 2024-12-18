import typing
from pathlib import Path

import polars as pl
import pytest

from biomarkers import imgs, utils
from biomarkers.models import bids, postprocess

confond_columns = {
    "trans_x",
    "trans_x_derivative1",
    "trans_x_power2",
    "trans_x_derivative1_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_power2",
    "trans_y_derivative1_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_power2",
    "trans_z_derivative1_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_power2",
    "rot_x_derivative1_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_power2",
    "rot_y_derivative1_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_power2",
    "rot_z_derivative1_power2",
    "0",
    "1",
    "2",
    "3",
    "4",
}


def test_confounds_flow(tmp_path: Path):
    sub = "travel2"
    ses = "RU"
    task = "rest"
    run = "01"
    space = "MNI152NLin6Asym"
    layout = bids.Layout.from_path(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep")
    )
    compcor_label = "WM+CSF"
    flow = postprocess.PostProcessRunFlow(
        dst=tmp_path,
        sub=sub,
        ses=ses,
        layout=layout,
        task=task,
        run=run,
        space=space,
        n_non_steady_state_tr=350,  # for speed
        compcor_label=compcor_label,
    )
    compcor = imgs.CompCor(
        img=flow.preproc,
        probseg=flow.probseg,
        label=compcor_label,
        boldref=flow.boldref,
        high_pass=flow.high_pass,
        low_pass=flow.low_pass,
        n_non_steady_state_tr=flow.n_non_steady_state_tr,
        detrend=flow.detrend,
    ).fit_transform()

    utils.update_confounds(
        flow.confounds,
        confounds=flow.layout.get_confounds(filters=flow.filter),
        n_non_steady_state_tr=flow.n_non_steady_state_tr,
        compcor=compcor,
    )

    confounds = pl.read_parquet(flow.confounds)
    all_there = [column in confounds.columns for column in confond_columns]
    nothing_extra = [column in confond_columns for column in confounds.columns]
    no_nulls = confounds.drop_nulls().shape[0] > 0

    assert all([all_there, nothing_extra, no_nulls])


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
