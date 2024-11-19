import typing
from pathlib import Path

import pytest

from biomarkers import imgs
from biomarkers.entrypoints import signatures, tapismpi
from biomarkers.flows import signature


@pytest.mark.parametrize("low_pass", [0.1, None])
@pytest.mark.parametrize("detrend", [True, False])
@pytest.mark.parametrize("winsorize", [True, False])
@pytest.mark.parametrize("compcor_label", typing.get_args(imgs.COMPCOR_LABEL))
def test_sig_flow(
    tmp_path: Path,
    low_pass: float | None,
    detrend: bool,
    winsorize: bool,
    compcor_label: imgs.COMPCOR_LABEL,
):
    signature.signature_flow(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep"),
        out=tmp_path,
        low_pass=low_pass,
        n_non_steady_state_tr=15,
        detrend=detrend,
        winsorize=winsorize,
        compcor_label=compcor_label,
    )
    exists = [
        (tmp_path / d).exists()
        for d in (
            "signatures-by-part",
            "signatures-by-run",
            "signatures-by-tr",
            "signatures-cleaned",
            "signatures-confounds",
        )
    ]
    assert all(exists)


def test_sig_pair_flow(tmp_path: Path):
    signature.signature_flow(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep"),
        out=tmp_path,
        low_pass=None,
        n_non_steady_state_tr=15,
        baseline_list=["rest1", "rest2"],
        active_list=["cuff1", "cuff2"],
    )
    exists = [
        (tmp_path / d).exists()
        for d in (
            "signatures-by-part",
            "signatures-by-run",
            "signatures-by-tr",
            "signatures-by-part-diff",
            "signatures-by-run-diff",
            "signatures-by-tr-diff",
            "signatures-cleaned",
            "signatures-confounds",
        )
    ]
    assert all(exists)


@pytest.mark.mpi
@pytest.mark.asyncio
async def test_sig_entrypoint(tmp_path: Path):
    tapismpi.configure_mpi_logger()
    out = tmp_path / "signatures"
    await signatures.SignatureEntrypoint(
        ins=[Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep")],
        outs=[out],
        low_pass=0.1,
        n_non_steady_state_tr=15,
        detrend=False,
        winsorize=False,
    ).run()

    assert len(list(out.glob("*tar"))) > 0
