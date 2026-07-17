from pathlib import Path

import pytest

from scripts.download_resisc45 import _download_verified, _load_external_dataset, _verify_split


def test_external_dataset_download_contract_is_https_and_noncommercial(project_root: Path) -> None:
    dataset = _load_external_dataset(
        project_root / "configs/evaluation/external_calibration_v1.json"
    )
    assert dataset["archive"]["url"].startswith("https://")
    assert dataset["archive"]["sha256"] == (
        "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
    )
    assert dataset["redistribution"]["stated_license"] == "CC-BY-NC-4.0"
    assert dataset["redistribution"]["commercial_use_permitted"] is False


def test_split_verification_rejects_duplicate_or_unsafe_rows(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("airplane_001.jpg\nairplane_001.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _verify_split(duplicate, 2)

    unsafe = tmp_path / "unsafe.txt"
    unsafe.write_text("../airplane_001.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        _verify_split(unsafe, 1)


def test_verified_download_reuses_matching_local_file(tmp_path: Path) -> None:
    destination = tmp_path / "existing.bin"
    destination.write_bytes(b"verified")
    downloaded = _download_verified(
        url="https://example.invalid/not-used",
        destination=destination,
        expected_sha256="1c34f88707b55e6104c4eb20e71ffa3d33e414b71ef689a15fad0640d0ac58cb",
        expected_size=8,
    )
    assert downloaded is False
