from __future__ import annotations

from pathlib import Path

import pydantic
from pydantic.dataclasses import dataclass

from biomarkers import utils


@dataclass(frozen=True)
class CATLabel:
    subdir: pydantic.DirectoryPath
    catROIs_mat: pydantic.FilePath
    catROIs_xml: pydantic.FilePath
    catROI_mat: pydantic.FilePath
    catROI_xml: pydantic.FilePath

    @classmethod
    def from_root(cls, root: Path, src: str) -> "CATLabel":
        subdir = root / "label"
        return cls(
            subdir=subdir,
            catROIs_mat=subdir / f"catROIs_{src}.mat",
            catROIs_xml=subdir / f"catROIs_{src}.xml",
            catROI_mat=subdir / f"catROI_{src}.mat",
            catROI_xml=subdir / f"catROI_{src}.xml",
        )


@dataclass(frozen=True)
class CATMRI:
    subdir: pydantic.DirectoryPath
    mwp1sub: pydantic.FilePath
    mwp2sub: pydantic.FilePath
    p0sub: pydantic.FilePath
    wmsub: pydantic.FilePath
    y_sub: pydantic.FilePath

    @classmethod
    def from_root(cls, root: Path, src: str) -> CATMRI:
        subdir = root / "mri"
        return cls(
            subdir=subdir,
            mwp1sub=(subdir / f"mwp1{src}").with_suffix(".nii"),
            mwp2sub=(subdir / f"mwp2{src}").with_suffix(".nii"),
            p0sub=(subdir / f"p0{src}").with_suffix(".nii"),
            wmsub=(subdir / f"wm{src}").with_suffix(".nii"),
            y_sub=(subdir / f"y_{src}").with_suffix(".nii"),
        )


@dataclass(frozen=True)
class CATReport:
    subdir: pydantic.DirectoryPath
    catlog: pydantic.FilePath
    catreportj: pydantic.FilePath
    catreport: pydantic.FilePath
    cat_mat: pydantic.FilePath
    cat_xml: pydantic.FilePath

    @classmethod
    def from_root(cls, root: Path, src: str) -> CATReport:
        subdir = root / "report"
        return cls(
            subdir=subdir,
            catlog=(subdir / f"catlog_{src}.nii.txt"),
            catreportj=(subdir / f"catreportj_{src}").with_suffix(".jpg"),
            catreport=(subdir / f"catreport_{src}").with_suffix(".pdf"),
            cat_mat=(subdir / f"cat_{src}").with_suffix(".mat"),
            cat_xml=(subdir / f"cat_{src}").with_suffix(".xml"),
        )


@dataclass(frozen=True)
class CATSurf:
    subdir: pydantic.DirectoryPath
    lh_central: pydantic.FilePath
    lh_pbt: pydantic.FilePath
    lh_pial: pydantic.FilePath
    lh_sphere_reg: pydantic.FilePath
    lh_sphere: pydantic.FilePath
    lh_thickness: pydantic.FilePath
    lh_white: pydantic.FilePath
    rh_central: pydantic.FilePath
    rh_pbt: pydantic.FilePath
    rh_pial: pydantic.FilePath
    rh_sphere_reg: pydantic.FilePath
    rh_sphere: pydantic.FilePath
    rh_thickness: pydantic.FilePath
    rh_white: pydantic.FilePath

    @classmethod
    def from_root(cls, root: Path, src: str) -> CATSurf:
        subdir = root / "surf"
        return cls(
            subdir=subdir,
            lh_central=(subdir / f"lh.central.{src}.gii"),
            lh_pbt=(subdir / f"lh.pbt.{src}"),
            lh_pial=(subdir / f"lh.pial.{src}.gii"),
            lh_sphere_reg=(subdir / f"lh.sphere.reg.{src}.gii"),
            lh_sphere=(subdir / f"lh.sphere.{src}.gii"),
            lh_thickness=subdir / f"lh.thickness.{src}",
            lh_white=(subdir / f"lh.white.{src}.gii"),
            rh_central=(subdir / f"rh.central.{src}.gii"),
            rh_pbt=(subdir / f"rh.pbt.{src}"),
            rh_pial=(subdir / f"rh.pial.{src}.gii"),
            rh_sphere_reg=(subdir / f"rh.sphere.reg.{src}.gii"),
            rh_sphere=(subdir / f"rh.sphere.{src}.gii"),
            rh_thickness=subdir / f"rh.thickness.{src}",
            rh_white=(subdir / f"rh.white.{src}.gii"),
        )


@dataclass(frozen=True)
class CATResult:
    root: pydantic.DirectoryPath
    img: pydantic.FilePath
    label: CATLabel
    mri: CATMRI
    report: CATReport
    surf: CATSurf

    @classmethod
    def from_root(cls, root: Path, img: Path) -> CATResult:
        src = utils.img_stem(img)
        return cls(
            root=root,
            img=img,
            label=CATLabel.from_root(root=root, src=src),
            mri=CATMRI.from_root(root=root, src=src),
            report=CATReport.from_root(root=root, src=src),
            surf=CATSurf.from_root(root=root, src=src),
        )
