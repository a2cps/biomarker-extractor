import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import polars as pl
import pydantic
from ancpbids import pybids_compat
from nibabel import processing
from nilearn import _utils, masking

PROBSEG_LABEL: typing.TypeAlias = typing.Literal["WM", "CSF", "GM"]


class Layout(pydantic.BaseModel):
    layout: pybids_compat.BIDSLayout
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @property
    def subjects(self) -> list[str]:
        return [str(sub) for sub in self.layout.get_subjects()]

    def get(self, filters: dict[str, str]) -> Path:
        file = self.layout.get(return_type="filename", **filters)
        if not len(file) == 1:
            msg = f"Expected that only 1 file would be retreived but saw {file=}; {filters=}"
            raise ValueError(msg)
        return Path(str(file[0]))

    def get_confounds(self, filters: dict[str, str]) -> Path:
        return self.get(filters={"desc": "confounds", "extension": ".tsv", **filters})

    def get_probseg(self, filters: dict[str, str], label: PROBSEG_LABEL) -> Path:
        return self.get(filters={"label": label, **filters})

    def get_wm(self, filters: dict[str, str]) -> Path:
        return self.get_probseg(filters=filters, label="WM")

    def get_gm(self, filters: dict[str, str]) -> Path:
        return self.get_probseg(filters=filters, label="GM")

    def get_csf(self, filters: dict[str, str]) -> Path:
        return self.get_probseg(filters=filters, label="CSF")

    def get_boldref(self, filters: dict[str, str]) -> Path:
        return self.get(
            filters={**filters, "suffix": "boldref", "extension": ".nii.gz"}
        )

    def get_brain(self, filters: dict[str, str]) -> Path:
        return self.get(
            filters={
                **filters,
                "desc": "brain",
                "suffix": "mask",
                "extension": ".nii.gz",
            }
        )

    def get_preproc(self, filters: dict[str, str]) -> Path:
        return self.get(
            filters={
                **filters,
                "suffix": "bold",
                "extension": ".nii.gz",
                "desc": "preproc",
            }
        )

    def get_sessions(self, sub: str) -> list[str]:
        if sub not in self.subjects:
            msg = f"Unable to find {sub} in layout"
            raise AssertionError(msg)

        return [str(ses) for ses in self.layout.get_sessions(sub=sub)]

    def get_tasks(self, sub: str, ses: str) -> list[str]:
        return [str(task) for task in self.layout.get_tasks(sub=sub, ses=ses)]

    def get_runs(self, sub: str, ses: str, task: str) -> list[str]:
        return [str(run) for run in self.layout.get_runs(sub=sub, ses=ses, task=task)]

    @classmethod
    def from_path(cls, src: Path) -> typing.Self:
        return cls(layout=pybids_compat.BIDSLayout(str(src)))


class ProbSeg(pydantic.BaseModel):
    GM: pydantic.FilePath
    WM: pydantic.FilePath
    CSF: pydantic.FilePath

    @property
    def gm_nii(self) -> nb.nifti1.Nifti1Image:
        return nb.nifti1.Nifti1Image.load(self.GM)

    @property
    def wm_nii(self) -> nb.nifti1.Nifti1Image:
        return nb.nifti1.Nifti1Image.load(self.WM)

    @property
    def csf_nii(self) -> nb.nifti1.Nifti1Image:
        return nb.nifti1.Nifti1Image.load(self.CSF)

    @classmethod
    def from_layout(cls, layout: Layout, filters: dict[str, str]) -> typing.Self:
        filters = {"suffix": "probseg", "extension": ".nii.gz", **filters}
        gm = layout.get_gm(filters=filters)
        wm = layout.get_wm(filters=filters)
        csf = layout.get_csf(filters=filters)
        return cls(GM=gm, WM=wm, CSF=csf)


class Func3d(pydantic.BaseModel):
    label: str
    path: pydantic.FilePath
    dtype: str = "f8"

    def to_polars(self, target: nb.nifti1.Nifti1Image | None = None) -> pl.DataFrame:
        i: nb.nifti1.Nifti1Image = _utils.check_niimg_3d(self.path)  # type: ignore

        if target:
            img = processing.resample_from_to(i, target, order=1)
        else:
            img = i

        out = masking.apply_mask(img, make_mask(img), dtype=self.dtype)
        return pl.DataFrame(
            {"voxel": np.arange(out.shape[0], dtype=np.uint32), self.label: out}
        )


class Func4d(pydantic.BaseModel):
    path: pydantic.FilePath
    dtype: str = "f8"
    label: str = "signal"

    def to_polars(self) -> pl.DataFrame:
        """Convert 4D nifti image into polars dataframe

        Returns:
            pl.DataFrame: data from input img
        """

        i: nb.nifti1.Nifti1Image = _utils.check_niimg(self.path, ensure_ndim=4)  # type: ignore
        return from_4d_to_polars(img=i, value_name=self.label, dtype=self.dtype)


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


def from_4d_to_polars(
    img: nb.nifti1.Nifti1Image, value_name: str = "signal", dtype: str = "f"
) -> pl.DataFrame:
    out = masking.apply_mask(img, make_mask(img), dtype=dtype)
    d = (
        pl.DataFrame(out, schema=[str(x) for x in range(out.shape[1])])
        .with_columns(pl.Series("t", np.arange(out.shape[0], dtype=np.uint16)))
        .unpivot(index=["t"], value_name=value_name, variable_name="voxel")
        .with_columns(pl.col("voxel").cast(pl.UInt32()))
    )

    return d
