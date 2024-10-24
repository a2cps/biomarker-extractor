import asyncio
import contextlib
import logging
import os
import re
import socket
import subprocess
import tempfile
import typing
from importlib import resources
from pathlib import Path
from typing import Callable, Concatenate, Iterable, Literal, ParamSpec, TypeVar

import nibabel as nb
import numpy as np
import pandas as pd
import polars as pl

FAILURE_LOG_DST = Path(os.environ.get("FAILURE_LOG_DST", "logs"))
DIR_PERMISSIONS = 0o750
FILE_PERMISSIONS = 0o640


def configure_root_logger() -> None:
    host = socket.gethostname()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | %(message)s",
        level=logging.INFO,
        force=True,
    )


def img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def get_mpfc_mask() -> Path:
    """Return mPFC mask produced from Smallwood et al. replication.

    Returns:
        Path: Path to mPFC mask.
    """
    with resources.path("biomarkers.data", "smallwood_mpfc_MNI152_1p5.nii.gz") as f:
        mpfc = f
    return mpfc


def get_rs2_labels() -> Path:
    with resources.path("biomarkers.data", "TD_label.nii") as f:
        labels = f
    return labels


def get_fan_atlas_file(resolution: Literal["2mm", "3mm"] = "2mm") -> Path:
    """Return file from ToPS model (https://doi.org/10.1038/s41591-020-1142-7)

    Returns:
        Path: Path to atlas
    """
    with resources.path(
        "biomarkers.data", f"Fan_et_al_atlas_r279_MNI_{resolution}.nii.gz"
    ) as f:
        atlas = f
    return atlas


def get_power2011_coordinates_file() -> Path:
    """Return file for volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        Path: Path to atlas
    """
    with resources.path("biomarkers.data", "power2011.tsv") as f:
        atlas = f
    return atlas


def get_power2011_coordinates() -> pd.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    return pd.read_csv(
        get_power2011_coordinates_file(),
        delim_whitespace=True,
        index_col="ROI",
        dtype={"x": np.float16, "y": np.int16, "z": np.int16},
    )


def get_mni6gray_mask() -> Path:
    with resources.path("biomarkers.data", "MNI152_T1_6mm_gray.nii.gz") as f:
        out = f
    return out


def exclude_to_index(n_non_steady_state_tr: int, n_tr: int) -> np.ndarray:
    return np.array([x for x in range(n_non_steady_state_tr, n_tr)])


def get_tr(nii: nb.nifti1.Nifti1Image) -> float:
    return nii.header.get("pixdim")[4]  # type: ignore


def get_nps_mask(
    weights: (Literal["negative", "positive", "rois", "group", "binary"] | None) = None,
) -> Path:
    match weights:
        case "negative":
            fname = "weights_NSF_negative_smoothed_larger_than_10vox.nii.gz"
        case "positive":
            fname = "weights_NSF_positive_smoothed_larger_than_10vox.nii.gz"
        case "rois":
            fname = "weights_NSF_smoothed_larger_than_10vox.nii.gz"
        case "group":
            fname = "weights_NSF_grouppred_cvpcr.nii.gz"
        case "binary":
            fname = "weights_NSF_grouppred_cvpcr_binary.nii.gz"
        case _:
            raise ValueError(f"{weights=} not recognized")

    with resources.path("biomarkers.data", fname) as f:
        path = f
    return path


P = ParamSpec("P")
R = TypeVar("R")


def cache_nii(
    f: Callable[P, nb.nifti1.Nifti1Image],
) -> Callable[Concatenate[Path, P], Path]:
    def wrapper(_filename: Path, *args: P.args, **kwargs: P.kwargs) -> Path:
        if _filename.exists():
            print(f"found cached {_filename}")
        else:
            out = f(*args, **kwargs)
            parent = _filename.parent
            if not parent.exists():
                parent.mkdir(parents=True)
            out.to_filename(_filename)
        return _filename

    # otherwise logging won't name of wrapped function
    # NOTE: unsure why @functools.wraps(f) doesn't work.
    # ends up complaining about the signature
    for attr in ("__name__", "__qualname__"):
        try:
            value = getattr(f, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)

    return wrapper


def cache_dataframe(
    f: Callable[P, pd.DataFrame | pl.DataFrame],
) -> Callable[Concatenate[Path | None, P], Path]:
    def wrapper(_filename: Path | None, *args: P.args, **kwargs: P.kwargs) -> Path:
        if _filename and _filename.exists():
            print(f"found cached {_filename}")
            outfile = _filename
        else:
            out = f(*args, **kwargs)
            if _filename:
                parent = _filename.parent
                if not parent.exists():
                    parent.mkdir(parents=True)
                outfile = _filename
            else:
                outfile = Path(tempfile.mkstemp(suffix=".parquet")[1])
            if isinstance(out, pd.DataFrame):
                out.columns = out.columns.astype(str)
                out.to_parquet(path=outfile, write_statistics=True)
            else:
                out.write_parquet(outfile, statistics=True)
        return outfile

    # otherwise logging won't name of wrapped function
    # NOTE: unsure why @functools.wraps(f) doesn't work.
    # ends up complaining about the signature
    for attr in ("__name__", "__qualname__"):
        try:
            value = getattr(f, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)

    return wrapper


def mat_to_df(cormat: np.ndarray, labels: Iterable[str]) -> pd.DataFrame:
    source = []
    target = []
    connectivity = []
    for xi, x in enumerate(labels):
        for yi, y in enumerate(labels):
            if yi <= xi:
                continue
            else:
                source.append(x)
                target.append(y)
                connectivity.append(cormat[xi, yi])

    return pd.DataFrame.from_dict(
        {"source": source, "target": target, "connectivity": connectivity}
    )


def get_poly_design(N: int, degree: int) -> np.ndarray:
    x = np.arange(N)
    x = x - np.mean(x, axis=0)
    X = np.vander(x, degree, increasing=True)
    q, r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    Z = raw / np.sqrt(norm2)
    return Z


def detrend(img: nb.nifti1.Nifti1Image, mask: Path) -> nb.nifti1.Nifti1Image:
    from nilearn import masking

    Y = masking.apply_mask(img, mask_img=mask)

    resid = _detrend(Y=Y)
    # Put results back into Niimg-like object
    return masking.unmask(resid, mask)  # type: ignore


def _detrend(Y: np.ndarray) -> np.ndarray:
    X = get_poly_design(Y.shape[0], degree=3)
    beta = np.linalg.pinv(X).dot(Y)
    return Y - np.dot(X[:, 1:], beta[1:, :])


def run_and_log_stdout(cmd: list[str], log: Path) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    log.write_text(out.stdout)
    return out.stdout


def mkdir_recursive(p: Path, mode: int = 0o770) -> None:
    for parent in reversed(p.parents):
        if not parent.exists():
            parent.mkdir(mode=mode)
    if not p.exists():
        p.mkdir(mode=mode)


def get_entity(f: Path, pattern: str) -> str:
    possibility = re.findall(pattern, str(f))
    if not len(possibility):
        raise ValueError
    return possibility[0]


def get_sub(f: Path) -> str:
    return get_entity(f=f, pattern=r"(?<=sub-)[a-zA-Z\d]+")


def get_ses(f: Path) -> str:
    return get_entity(f=f, pattern=r"(?<=ses-)[a-zA-Z\d]+")


def get_sub_from_sublong(f: Path) -> str:
    return get_entity(f, r"\d{5}")


def get_ses_from_sublong(f: Path) -> str:
    return get_entity(f, r"V[13]")


def compare_arg_lengths(
    to_check: typing.Sequence[bool] | None,
    to_compare: typing.Sequence[Path],
    arg_names: tuple[str, str],
) -> typing.Sequence[bool]:
    if to_check is None:
        out = [False] * len(to_compare)
    else:
        if not len(to_check) == len(to_compare):
            msg = f"""
            If {arg_names[0]} is provided, it must have the same lengths as {arg_names[1]}.
            Found len(images)={len(to_check)} and len(precrops)={len(to_compare)}
            """
            raise AssertionError(msg)
        out = to_check

    return out


@contextlib.asynccontextmanager
async def subprocess_manager(
    log: Path, args: list[str]
) -> typing.AsyncIterator[asyncio.subprocess.Process]:
    logging.info(f"{args=}")

    with open(log, mode="w") as stdout:
        procs = await asyncio.create_subprocess_exec(
            *args, stderr=subprocess.STDOUT, stdout=stdout
        )
        try:
            yield procs
        finally:
            if procs.returncode is None:
                procs.terminate()


def recursive_chmod(path: Path, file_mode=FILE_PERMISSIONS, dir_mode=DIR_PERMISSIONS):
    # Apply chmod to the current directory
    path.chmod(dir_mode)

    # Traverse subdirectories and files
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            (Path(root) / dir_name).chmod(dir_mode)

        for file_name in files:
            (Path(root) / file_name).chmod(file_mode)
