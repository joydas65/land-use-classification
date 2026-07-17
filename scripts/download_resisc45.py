"""Download and safely extract the checksum-pinned RESISC45 calibration dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.download_dataset import safe_extract, sha256


def _load_external_dataset(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("external calibration schema_version must be 1")
    dataset = config["external_dataset"]
    archive = dataset["archive"]
    if urlparse(archive["url"]).scheme != "https":
        raise ValueError("RESISC45 archive URL must use HTTPS")
    for split in dataset["splits"].values():
        if urlparse(split["url"]).scheme != "https":
            raise ValueError("RESISC45 split URLs must use HTTPS")
    return dataset


def _download_verified(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int | None = None,
) -> bool:
    if destination.is_file():
        size_matches = expected_size is None or destination.stat().st_size == expected_size
        if size_matches and sha256(destination) == expected_sha256:
            return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TerraClass/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise ValueError(
                f"Download size mismatch for {destination.name}: "
                f"expected {expected_size}, got {partial.stat().st_size}"
            )
        actual_sha256 = sha256(partial)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Download SHA-256 mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return True


def _verify_split(path: Path, expected_rows: int) -> None:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name} contains {len(rows)} rows; expected {expected_rows}")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{path.name} contains duplicate filenames")
    if any(Path(row).name != row or not row.lower().endswith(".jpg") for row in rows):
        raise ValueError(f"{path.name} contains an unsafe or unexpected filename")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/external_calibration_v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    dataset = _load_external_dataset(config_path)
    archive = dataset["archive"]
    archive_path = root / archive["path"]
    archive_downloaded = _download_verified(
        url=archive["url"],
        destination=archive_path,
        expected_sha256=archive["sha256"],
        expected_size=int(archive["size_bytes"]),
    )
    split_results: dict[str, dict[str, Any]] = {}
    for split_name, split in dataset["splits"].items():
        split_path = root / split["path"]
        downloaded = _download_verified(
            url=split["url"],
            destination=split_path,
            expected_sha256=split["sha256"],
        )
        _verify_split(split_path, int(split["rows"]))
        split_results[split_name] = {
            "path": split["path"],
            "sha256": sha256(split_path),
            "rows": int(split["rows"]),
            "downloaded": downloaded,
        }

    dataset_root = root / dataset["root"]
    if not args.skip_extraction and not dataset_root.is_dir():
        with zipfile.ZipFile(archive_path) as archive_file:
            safe_extract(archive_file, dataset_root.parent)
    if not args.skip_extraction:
        class_directories = sorted(path for path in dataset_root.iterdir() if path.is_dir())
        image_count = sum(
            1 for class_directory in class_directories for _ in class_directory.glob("*.jpg")
        )
        if len(class_directories) != 45 or image_count != 31_500:
            raise ValueError(
                "Extracted RESISC45 structure is invalid: "
                f"{len(class_directories)} classes and {image_count} images"
            )

    metadata = {
        "schema_version": 1,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset["name"],
        "archive": {
            "path": archive["path"],
            "sha256": sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "downloaded": archive_downloaded,
        },
        "splits": split_results,
        "extracted": not args.skip_extraction,
        "dataset_root": dataset["root"],
        "license": dataset["redistribution"]["stated_license"],
    }
    metadata_path = root / "data/raw/RESISC45_DOWNLOAD_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
