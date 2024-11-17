import asyncio
import contextlib
import gzip
import logging
import os
import re
import socket
import subprocess
import tempfile
import typing
from pathlib import Path
from typing import ParamSpec, TypeVar

import nibabel as nb
import numpy as np
import pandas as pd
import polars as pl
from nibabel import processing
from nilearn import _utils, masking

from biomarkers.models import signatures

P = ParamSpec("P")
R = TypeVar("R")

FAILURE_LOG_DST = Path(os.environ.get("FAILURE_LOG_DST", "logs"))
DIR_PERMISSIONS = 0o40750
FILE_PERMISSIONS = 0o100640


def configure_root_logger() -> None:
    host = socket.gethostname()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | %(message)s",
        level=logging.INFO,
        force=True,
    )


def img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def exclude_to_index(n_non_steady_state_tr: int, n_tr: int) -> np.ndarray:
    return np.array([x for x in range(n_non_steady_state_tr, n_tr)])


def get_tr(nii: nb.nifti1.Nifti1Image) -> float:
    return nii.header.get("pixdim")[4]  # type: ignore


def cache_nii(
    f: typing.Callable[P, nb.nifti1.Nifti1Image],
) -> typing.Callable[typing.Concatenate[Path, P], Path]:
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
    f: typing.Callable[P, pd.DataFrame | pl.DataFrame],
) -> typing.Callable[typing.Concatenate[Path | None, P], Path]:
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


def mat_to_df(cormat: np.ndarray, labels: typing.Iterable[str]) -> pd.DataFrame:
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


def mkdir_recursive(p: Path, mode: int = DIR_PERMISSIONS) -> None:
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


def gzip_file(src: Path, dst: Path):
    with open(src, "rb") as s:
        with gzip.open(dst, "wb") as d:
            for chunk in iter(lambda: s.read(4096), b""):
                d.write(chunk)


def make_mask(img: nb.nifti1.Nifti1Image) -> nb.nifti1.Nifti1Image:
    """Make mask that can be used as no-op.

    Args:
        img nb.Nifti1Image: Image whose shape and affine will be used to make mask.

    Returns:
        nb.Nifti1Image: Image (uint8) of size input with all values equal to 1
    """
    return nb.nifti1.Nifti1Image(
        dataobj=np.ones(img.shape[:3], dtype=np.uint8), affine=img.affine
    )


def to_df3d(img: nb.nifti1.Nifti1Image | Path, label: str, dtype="f8") -> pl.DataFrame:
    i: nb.nifti1.Nifti1Image = _utils.check_niimg_3d(img)  # type: ignore
    out = masking.apply_mask(i, make_mask(i), dtype=dtype)
    return pl.DataFrame({"voxel": np.arange(out.shape[0], dtype=np.uint32), label: out})


def to_df4d(img: nb.nifti1.Nifti1Image | Path, dtype: str = "f8") -> pl.DataFrame:
    """Convert 4D nifti image into polars dataframe

    Args:
        img: 4D nifti image
        dtype: datatype for value column. Defaults to "f8".

    Returns:
        pl.DataFrame: data from input img
    """

    i: nb.nifti1.Nifti1Image = _utils.check_niimg(img, ensure_ndim=4)  # type: ignore
    out = masking.apply_mask(i, make_mask(i), dtype=dtype)
    d = (
        pl.DataFrame(out, schema=[str(x) for x in range(out.shape[1])])
        .with_columns(pl.Series("t", np.arange(out.shape[0], dtype=np.int16)))
        .melt(id_vars=["t"], value_name="signal", variable_name="voxel")
        .with_columns(pl.col("voxel").cast(pl.UInt32()))
    )

    return d


@cache_dataframe
def to_parquet3d(
    fmriprep_func3ds: list[signatures.Func3d],
    func3ds: list[signatures.Func3d] | None = None,
) -> pl.DataFrame:
    if not len(fmriprep_func3ds):
        msg = "there should be at least 1 image to process"
        raise AssertionError(msg)

    d = to_df3d(
        fmriprep_func3ds[0].path,
        label=fmriprep_func3ds[0].label,
        dtype=fmriprep_func3ds[0].dtype,
    )
    for func3d in fmriprep_func3ds[1:]:
        d = d.join(
            to_df3d(
                func3d.path,
                label=func3d.label,
                dtype=func3d.dtype,
            ),
            on="voxel",
        )
    if func3ds:
        target = nb.loadsave.load(fmriprep_func3ds[0].path)
        for func3d in func3ds:
            img = nb.loadsave.load(func3d.path)
            d = d.join(
                to_df3d(
                    processing.resample_from_to(img, target, order=1),
                    label=func3d.label,
                    dtype=func3d.dtype,
                ),
                on="voxel",
            )
    return d


@cache_dataframe
def update_confounds(
    confounds: Path,
    n_non_steady_state_tr: int = 0,
    acompcor_file: Path | None = None,
    usecols: list[str] = [
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
    ],
    label: typing.Literal["CSF", "WM", "WM+CSF"] | None = "WM+CSF",
    extra: Path | None = None,
) -> pd.DataFrame:
    confounds_df = pd.read_csv(confounds, delim_whitespace=True, usecols=usecols)
    n_tr = confounds_df.shape[0] - n_non_steady_state_tr
    components_df = confounds_df.iloc[-n_tr:, :].reset_index(drop=True)
    if extra:
        extra_cols = pd.read_parquet(extra)
        components_df = pd.concat([components_df, extra_cols], axis=1)
    if acompcor_file and label:
        acompcor = pd.read_parquet(
            acompcor_file, columns=["component", "tr", "value", "label"]
        )
        components = (
            acompcor.query("label==@label and component < 5")
            .drop("label", axis=1)
            .pivot(index="tr", columns=["component"], values="value")
        )
        out = pd.concat([components_df, components], axis=1)
    else:
        out = components_df
    out.columns = out.columns.astype(str)
    return out
