import shutil
import tempfile
from pathlib import Path

from biomarkers import utils
from biomarkers.entrypoints import tapismpi


def get_eddy_args(bidsdir: Path, workdir: Path, outdir: Path) -> list[str]:
    qsiprep_wf = workdir / "qsiprep_wf"
    dwi_preproc_ses = None
    for d in qsiprep_wf.glob("single_subject_*_wf/dwi_preproc_ses_*_wf"):
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
        "eddy_quad",
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
    fs_license_file: Path
    eddy_params: Path
    n_workers: int | None = None
    mem_mb: int | None = None
    output_resolution: float = 1.7
    hmc_model: str = "eddy"
    unringing_method: str = "mrdegibbs"
    denoise_method: str = "patch2self"

    def check_outputs(self, output_dir_to_check: Path) -> bool:
        return output_dir_to_check.exists() and (
            len(list((output_dir_to_check / "qsiprep").glob("*html"))) > 0
        )

    def get_args(self, bidsdir: Path, outdir: Path, work_dir: Path) -> list[str]:
        args = ["qsiprep", "--notrack", "--skip-bids-validation"]

        to_extend = {
            "--output-resolution": self.output_resolution,
            "--fs-license-file": self.fs_license_file,
            "--hmc_model": self.hmc_model,
            "--unringing-method": self.unringing_method,
            "--denoise-method": self.denoise_method,
            "--nthreads": self.n_workers,
            "--mem_mb": self.mem_mb,
            "--eddy-config": self.eddy_params,
            "--work-dir": work_dir,
        }
        for key, value in to_extend.items():
            extend_arg(args, key, value)

        args.extend([str(bidsdir), str(outdir / "qsiprep"), "participant"])

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
