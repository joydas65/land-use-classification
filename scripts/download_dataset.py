"""Download and safely extract the UC Merced dataset used by the baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_5class.json"))
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", choices=("mirror", "original"), default="mirror")
    parser.add_argument("--allow-insecure-http", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    url = dataset["verified_mirror_url"] if args.source == "mirror" else dataset["source_url"]
    if urlparse(url).scheme != "https" and not args.allow_insecure_http:
        raise SystemExit(
            "The original source uses unencrypted HTTP. Review the URL, then rerun with "
            "--allow-insecure-http or replace it with a verified HTTPS mirror."
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    archive_path = args.destination / "UCMerced_LandUse.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "TerraClass/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    archive_hash = sha256(archive_path)
    if archive_hash != dataset["archive_sha256"]:
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {dataset['archive_sha256']}, got {archive_hash}"
        )
    if archive_path.stat().st_size != dataset["archive_size_bytes"]:
        raise ValueError(
            f"Archive size mismatch: expected {dataset['archive_size_bytes']}, "
            f"got {archive_path.stat().st_size}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        safe_extract(archive, args.destination)
    metadata = {
        "schema_version": 1,
        "source_kind": args.source,
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive_path.stat().st_size,
    }
    (args.destination / "DOWNLOAD_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
