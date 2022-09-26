from typing import Literal
from pathlib import Path

import pydantic
from pydantic.dataclasses import dataclass


def _from_entities(
    root: Path,
    sub: str,
    suffix: str,
    extension: str,
    modality: str,
    ses: str | None = None,
    fromto: tuple(str, str) | None = None,
    mode: str | None = None,
    label: str | None = None,
    desc: str | None = None,
    space: str | None = None,
) -> Path:
    folder = root / f"sub-{sub}"
    name = f"sub-{sub}"
    if ses:
        folder = folder / f"ses-{ses}"
        name += f"_ses-{ses}"
    folder = folder / modality
    if fromto:
        name += f"_from-{fromto[0]}_to-{fromto[1]}"
    if mode:
        name += f"_mode-{mode}"
    if space:
        name += f"_space-{space}"
    if label:
        name += f"_label-{label}"
    if desc:
        name += f"_desc-{desc}"
    name += f"_{suffix}"
    path = (folder / name).with_suffix(extension)
    return path


@dataclass(frozen=True)
class File:
    stem: str
    sub: str
    suffix: str
    extension: str
    modality: Literal["anat", "func", "fmap", "dwi"]
    path: pydantic.FilePath
    ses: str | None
    desc: str | None
    space: str | None
    label: str | None

    @classmethod
    def from_entities(
        cls,
        root: Path,
        sub: str,
        suffix: str,
        extension: str,
        modality: str,
        ses: str | None = None,
        label: str | None = None,
        desc: str | None = None,
        space: str | None = None,
    ) -> "File":
        path = _from_entities(
            root=root,
            sub=sub,
            suffix=suffix,
            extension=extension,
            modality=modality,
            ses=ses,
            label=label,
            desc=desc,
            space=space,
        )
        return cls(
            stem=path.stem,
            path=path,
            extension=extension,
            modality=modality,
            sub=sub,
            suffix=suffix,
            label=label,
            ses=ses,
            desc=desc,
            space=space,
        )


@dataclass(frozen=True)
class Transform(File):
    fromto: tuple(str)
    mode: str

    @classmethod
    def from_entities(
        cls,
        root: Path,
        sub: str,
        extension: str,
        fromto: tuple(str, str),
        modality: str,
        ses: str | None = None,
        label: str | None = None,
        desc: str | None = None,
        space: str | None = None,
        mode: str | None = None,
    ) -> "Transform":
        path = _from_entities(
            root=root,
            sub=sub,
            suffix="xfm",
            extension=extension,
            modality=modality,
            ses=ses,
            label=label,
            desc=desc,
            space=space,
            fromto=fromto,
            mode=mode,
        )
        return cls(
            stem=path.stem,
            path=path,
            sub=sub,
            suffix="xfm",
            extension=extension,
            modality=modality,
            ses=ses,
            label=label,
            desc=desc,
            space=space,
            fromto=fromto,
            mode=mode,
        )


@dataclass(frozen=True)
class Pair(File):
    meta: File

    @classmethod
    def from_entities(
        cls,
        root: Path,
        sub: str,
        suffix: str,
        extension: str,
        modality: str,
        ses: str | None = None,
        label: str | None = None,
        desc: str | None = None,
        space: str | None = None,
    ) -> "Pair":
        path = _from_entities(
            root=root,
            sub=sub,
            suffix=suffix,
            extension=extension,
            modality=modality,
            ses=ses,
            label=label,
            desc=desc,
            space=space,
        )
        meta = File.from_entities(
            root=root,
            sub=sub,
            suffix=suffix,
            extension="json",
            modality=modality,
            ses=ses,
            label=label,
            desc=desc,
            space=space,
        )
        return cls(
            stem=path.stem,
            path=path,
            extension=extension,
            modality=modality,
            sub=sub,
            suffix=suffix,
            label=label,
            ses=ses,
            desc=desc,
            space=space,
            meta=meta,
        )
