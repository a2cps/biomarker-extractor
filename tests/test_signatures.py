from pathlib import Path

import pytest

from biomarkers.entrypoints import signatures, tapismpi
from biomarkers.flows import signature


def test_sig_flow(tmp_path: Path):
    signature.signature_flow(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep"),
        out=tmp_path,
        low_pass=0.1,
        n_non_steady_state_tr=15,
        detrend=False,
        winsorize=False,
    )
    exists = [
        (tmp_path / d).exists()
        for d in (
            "signatures-by-part",
            "signatures-by-run",
            "signatures-by-tr",
            "cleaned",
            "confounds",
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
            "cleaned",
            "confounds",
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
