import typing

FORCEABLE: typing.TypeAlias = typing.Literal[
    "gradwarp1D",
    "gradwarp3D",
    "sdc-anat-reference",
    "jacobian",
    "gre-sdc-after-eddy",
    "no-csf-synthstrip",
]
