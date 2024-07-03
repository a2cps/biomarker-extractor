import typing

CIFTI_OUTPUT: typing.TypeAlias = typing.Literal["91k", "170k"]
OUTPUT_SPACE: typing.TypeAlias = typing.Literal[
    "MNI152NLin2009cAsym:res-native:res-2", "MNI152NLin6Asym:res-native:res-2"
]
BOLD2ANAT_DOF = typing.Literal[6, 9, 12]
