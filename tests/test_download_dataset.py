import zipfile
from pathlib import Path

import pytest

from scripts.download_dataset import safe_extract


def test_safe_extract_accepts_normal_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "normal.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/class/image.tif", b"image")
    destination = tmp_path / "output"
    destination.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        safe_extract(archive, destination)
    assert (destination / "dataset/class/image.tif").read_bytes() == b"image"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", b"unsafe")
    destination = tmp_path / "output"
    destination.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe archive member"):
            safe_extract(archive, destination)
