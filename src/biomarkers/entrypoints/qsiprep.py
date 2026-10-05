import shutil
import tempfile
import typing
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi
from biomarkers.models import qsiprep as qsiprep_models

# qsiprep's container image installs qsiprep (and eddy_quad) into a pixi env.
# the env's bin is on PATH via ENV, so no shell-hook is needed (and running
# `bash /shell-hook.sh <cmd>` would not exec <cmd>)
QSIPREP_BIN = Path("/app/.pixi/envs/qsiprep/bin")


def get_eddy_args(bidsdir: Path, workdir: Path, outdir: Path) -> list[str]:
    dwi_preproc_ses = None
    # e.g., qsiprep_26_1_wf/sub_01_ses_V1_wf/dwi_preproc_ses_V1_wf
    for d in workdir.glob("qsiprep_*_wf/sub_*_wf/dwi_preproc_*_wf"):
        dwi_preproc_ses = d
        break
    if dwi_preproc_ses is None:
        raise AssertionError("Unable to find qsiprep_wf! eddyqc will fail")
    bvals = None
    for bv in bidsdir.glob("sub*/ses*/dwi/*bval"):
        bvals = bv
        break
    if bvals is None:
        raise AssertionError("Unable to find bvals in bidsdir! eddyqc will fail")

    hmc_sdc_wf = dwi_preproc_ses / "hmc_sdc_wf"
    basename = hmc_sdc_wf / "eddy" / "eddy_corrected"
    idx = hmc_sdc_wf / "gather_inputs" / "eddy_index.txt"
    par = hmc_sdc_wf / "gather_inputs" / "eddy_acqp.txt"
    mask = (
        hmc_sdc_wf
        / "pre_eddy_b0_ref_wf"
        / "synthstrip_wf"
        / "mask_to_original_grid"
        / "topup_imain_corrected_avg_trans_mask_trans.nii.gz"
    )
    fieldmap = hmc_sdc_wf / "topup" / "fieldmap_HZ.nii.gz"
    args = [
        QSIPREP_BIN / "eddy_quad",
        basename,
        "-v",
        "-idx",
        idx,
        "-par",
        par,
        "-m",
        mask,
        "-b",
        bvals,
        "-f",
        fieldmap,
        "-o",
        outdir / "eddyqc",
    ]
    return [str(i) for i in args]


def extend_arg(
    args: list[str],
    name: str,
    value: str | int | bool | Path | None = None,
):
    if value:
        match name:
            case "--output-spaces":
                for space in str(value).split(" "):
                    args.extend([name, str(space)])
            case _:
                args.extend([name, str(value)])


class QSIPRepEntrypoint(tapismpi.TapisMPIEntrypoint):
    eddy_params: Path
    n_workers: int | None = None
    mem_mb: int | None = None
    output_resolution: float = 1.7
    hmc_method: str = "eddy"
    unringing_method: str = "mrdegibbs"
    denoise_method: str = "patch2self"
    force: typing.Sequence[qsiprep_models.FORCEABLE] | None = ("no-csf-synthstrip",)

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list((output_dir_to_check / "qsiprep").glob("*html"))) > 0
        )

    def get_args(self, bidsdir: Path, outdir: Path, work_dir: Path) -> list[str]:
        # positionals go first, because options like --force take nargs="+"
        # and would otherwise consume them
        # qsiprep writes directly into the output dir, so give it a subfolder
        args = [
            str(QSIPREP_BIN / "qsiprep"),
            str(bidsdir),
            str(outdir / "qsiprep"),
            "participant",
            "--notrack",
            "--skip-bids-validation",
        ]
        if self.force:
            for f in self.force:
                extend_arg(args, "--force", f)

        to_extend = {
            "--output-resolution": self.output_resolution,
            "--hmc-method": self.hmc_method,
            "--unringing-method": self.unringing_method,
            "--denoise-method": self.denoise_method,
            "--nthreads": self.n_workers,
            "--mem-mb": self.mem_mb,
            "--eddy-config": self.eddy_params,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        return args

    async def run_flow(self, tmpd_in: Path, tmpd_out: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpd:
            work_dir = Path(tmpd)
            async with utils.subprocess_manager(
                log=tmpd_out / f"qsiprep_rank-{self.RANK}.log",
                args=self.get_args(bidsdir=tmpd_in, outdir=tmpd_out, work_dir=work_dir),
            ) as proc:
                await proc.wait()
                if proc.returncode and proc.returncode > 0:
                    # remove folder so that archiving detects that there was a failure
                    # and sends logs to failure_dst_dir
                    if (outdir := tmpd_out / "qsiprep").exists():
                        shutil.rmtree(outdir)
                    msg = f"qsiprep failed with {proc.returncode=}"
                    raise RuntimeError(msg)

            async with utils.subprocess_manager(
                log=tmpd_out / f"eddyqc_rank-{self.RANK}.log",
                args=get_eddy_args(
                    tmpd_in, workdir=work_dir, outdir=tmpd_out / "eddyqc"
                ),
            ) as proc:
                await proc.wait()
                if proc.returncode and proc.returncode > 0:
                    # remove folder so that archiving detects that there was a failure
                    # and sends logs to failure_dst_dir
                    if (outdir := tmpd_out / "eddyqc").exists():
                        shutil.rmtree(outdir)
                    msg = f"eddyqc failed with {proc.returncode=}"
                    raise RuntimeError(msg)
