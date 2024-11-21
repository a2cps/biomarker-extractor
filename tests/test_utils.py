import os
import stat
from pathlib import Path

from biomarkers import utils


def get_permissions(src: Path) -> int:
    return stat.S_IMODE(src.stat().st_mode)


def test_img_stem_niigz():
    src = Path("img.nii.gz")
    assert utils.img_stem(src) == "img"


def test_img_stem_nii():
    src = Path("img.nii")
    assert utils.img_stem(src) == "img"


def test_gzip(tmp_path: Path):
    src = tmp_path / "togzip"
    src.write_text("content", encoding="utf-8")
    dst = tmp_path / "togzip.gz"
    utils.gzip_file(src, dst)
    assert dst.exists()


def test_recursive_chmod(tmp_path: Path):
    src = tmp_path / "subdir"
    src.mkdir()
    (src / "file.txt").touch()
    utils.recursive_chmod(tmp_path)
    files = []
    dirs = []
    for dirpath, dirnames, filenames in os.walk(tmp_path):
        root = Path(dirpath)
        files.extend(
            [get_permissions(root / f) == utils.FILE_PERMISSIONS for f in filenames]
        )
        dirs.extend(
            [get_permissions(root / d) == utils.DIR_PERMISSIONS for d in dirnames]
        )

    assert all(files + dirs)


def test_recursive_mkdir(tmp_path: Path):
    dst = tmp_path / "first" / "second" / "third"
    utils.mkdir_recursive(dst)
    assert dst.exists()
