from pathlib import Path

import pytest

# entrypoints import mpi4py, which needs an MPI installation
pytest.importorskip("mpi4py")

from biomarkers.entrypoints import qsiprep


def get_entrypoint(**kwargs) -> qsiprep.QSIPRepEntrypoint:
    return qsiprep.QSIPRepEntrypoint(
        ins=[Path("in")], outs=[Path("out")], eddy_params=Path("eddy.json"), **kwargs
    )


def test_get_args_positionals_first():
    args = get_entrypoint().get_args(
        bidsdir=Path("/bids"), outdir=Path("/out"), work_dir=Path("/work")
    )
    assert args[:4] == [
        str(qsiprep.QSIPREP_BIN / "qsiprep"),
        "/bids",
        "/out/qsiprep",
        "participant",
    ]


def test_get_args_force():
    args = get_entrypoint(force=["no-csf-synthstrip", "jacobian"]).get_args(
        bidsdir=Path("/bids"), outdir=Path("/out"), work_dir=Path("/work")
    )
    i = args.index("--force")
    assert args[i : i + 4] == ["--force", "no-csf-synthstrip", "--force", "jacobian"]


def test_get_args_removed_options():
    args = get_entrypoint(mem_mb=1000).get_args(
        bidsdir=Path("/bids"), outdir=Path("/out"), work_dir=Path("/work")
    )
    assert "--fs-license-file" not in args
    assert "--mem_mb" not in args
    assert args[args.index("--mem-mb") + 1] == "1000"


def test_get_eddy_args(tmp_path: Path):
    bidsdir = tmp_path / "bids"
    dwi = bidsdir / "sub-01" / "ses-V1" / "dwi"
    dwi.mkdir(parents=True)
    bval = dwi / "sub-01_ses-V1_dwi.bval"
    bval.touch()
    hmc_sdc_wf = (
        tmp_path
        / "work"
        / "qsiprep_26_1_wf"
        / "sub_01_ses_V1_wf"
        / "dwi_preproc_ses_V1_wf"
        / "hmc_sdc_wf"
    )
    hmc_sdc_wf.mkdir(parents=True)

    args = qsiprep.get_eddy_args(
        bidsdir, workdir=tmp_path / "work", outdir=tmp_path / "out"
    )

    assert args[0] == str(qsiprep.QSIPREP_BIN / "eddy_quad")
    assert args[1] == str(hmc_sdc_wf / "eddy" / "eddy_corrected")
    assert args[args.index("-idx") + 1] == str(
        hmc_sdc_wf / "gather_inputs" / "eddy_index.txt"
    )
    assert args[args.index("-f") + 1] == str(
        hmc_sdc_wf / "topup" / "fieldmap_HZ.nii.gz"
    )
    assert args[args.index("-b") + 1] == str(bval)
