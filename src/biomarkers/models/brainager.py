from pathlib import Path

import pydantic


class BrainAgeResult(pydantic.BaseModel):
    volumes: pydantic.FilePath
    predictions: pydantic.FilePath
    slices: pydantic.DirectoryPath

    @classmethod
    def from_nii(cls, nii: Path) -> "BrainAgeResult":
        return cls(
            volumes=nii.with_name(f"{nii.stem}_tissue_volumes.csv"),
            predictions=nii.with_suffix(".csv"),
            slices=nii.with_name(f"slicesdir_{nii.name}"),
        )
