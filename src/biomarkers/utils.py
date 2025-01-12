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

P = ParamSpec("P")
R = TypeVar("R")

FAILURE_LOG_DST = Path(os.environ.get("FAILURE_LOG_DST", "logs"))
DIR_PERMISSIONS = 0o750
FILE_PERMISSIONS = 0o640
MOTION_PARAMETERS: typing.TypeAlias = typing.Literal[
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
]
FRISTON_24: tuple[MOTION_PARAMETERS, ...] = typing.get_args(MOTION_PARAMETERS)


def configure_root_logger() -> None:
    host = socket.gethostname()
    logging.basicConfig(
        format=f"%(asctime)s | %(levelname)-8s | {host=} | %(message)s",
        level=logging.INFO,
        force=True,
    )


def img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def int_sample_mask_to_bool(sample_mask: np.typing.NDArray[np.uint32]) -> list[bool]:
    min_tr = sample_mask.min()
    n_tr = len(sample_mask)
    n_excluded_tr = len(np.arange(0, min_tr))
    return [False] * n_excluded_tr + [True] * n_tr


def exclude_to_index(
    n_non_steady_state_tr: int, n_tr: int
) -> np.typing.NDArray[np.uint32]:
    return np.array(
        [x for x in np.arange(n_non_steady_state_tr, n_tr, dtype=np.uint32)]
    )


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


@cache_dataframe
def update_confounds(
    confounds: Path,
    n_non_steady_state_tr: int = 0,
    usecols: typing.Sequence[str] = FRISTON_24,
    compcor: pl.DataFrame | None = None,
) -> pl.DataFrame:
    out = (
        pl.read_csv(confounds, separator="\t", columns=usecols, null_values="n/a")
        .with_row_index("t")
        .slice(n_non_steady_state_tr)
    )
    if compcor is not None:
        # right join because of slice above
        out = (
            compcor.filter(pl.col("component") < 5)
            .pivot(on="component", index="t", values="value")
            .join(out, how="right", on="t")
        )

    return out.sort("t").drop("t")


def write_parquet(
    d: pl.DataFrame, dst: Path, partition_by: typing.Sequence[str] | None = None
):
    if not (parent := dst.parent).exists():
        parent.mkdir(parents=True)

    d.write_parquet(file=dst, partition_by=partition_by)


def check_matching_image_shapes(imgs: typing.Sequence[Path]) -> bool:
    loaded_imgs = [nb.nifti1.Nifti1Image.load(i) for i in imgs]
    return all(
        [
            np.allclose(x.shape, y.shape)
            for x, y in zip(loaded_imgs[:-1], loaded_imgs[1:])
        ]
    )


def mat_to_df(cormat: np.ndarray, labels: typing.Sequence[int]) -> pl.DataFrame:
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

    return pl.DataFrame(
        {"source": source, "target": target, "connectivity": connectivity}
    )
