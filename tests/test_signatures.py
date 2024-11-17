from pathlib import Path

from biomarkers.flows import signature


def test_sig_entrypoint(tmp_path: Path):
    signature.signature_flow(
        Path("/Users/psadil/git/a2cps/biomarkers/tests/data/fmriprep"),
        out=tmp_path,
        low_pass=0.1,
        n_non_steady_state_tr=12,
        detrend=True,
        winsorize=False,
    )
    exists = [
        (tmp_path / d).exists()
        for d in (
            "signature-by-part",
            "signature-by-run",
            "signature-by-tr",
            "signature-cleaned",
            "signature-confounds",
            "signature-labels",
        )
    ]
    assert all(exists)
