"""Checksum-pinned distribution for the versioned serving artifact."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelRelease:
    schema_version: int
    model_id: str
    model_version: str
    asset_name: str
    url: str
    sha256: str
    size_bytes: int

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"unsupported schema_version={self.schema_version}")
        if not self.model_id.strip() or not self.model_version.strip():
            errors.append("model_id and model_version must not be blank")
        if Path(self.asset_name).name != self.asset_name or not self.asset_name:
            errors.append("asset_name must be a plain file name")
        if urlparse(self.url).scheme != "https":
            errors.append("url must use HTTPS")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            errors.append("sha256 must be a lowercase SHA-256")
        if self.size_bytes <= 0:
            errors.append("size_bytes must be positive")
        if errors:
            raise ValueError("Invalid model release: " + "; ".join(errors))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_release(path: str | Path) -> ModelRelease:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    release = ModelRelease(
        schema_version=int(raw["schema_version"]),
        model_id=str(raw["model_id"]),
        model_version=str(raw["model_version"]),
        asset_name=str(raw["asset_name"]),
        url=str(raw["url"]),
        sha256=str(raw["sha256"]),
        size_bytes=int(raw["size_bytes"]),
    )
    release.validate()
    return release


def fetch_release_artifact(
    release: ModelRelease,
    destination: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> str:
    """Download atomically and accept only the configured byte count and checksum."""
    release.validate()
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    destination = Path(destination)
    if destination.is_file():
        expected_file = destination.stat().st_size == release.size_bytes
        if expected_file and sha256(destination) == release.sha256:
            return release.sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        release.url,
        headers={"Accept": "application/octet-stream", "User-Agent": "TerraClass/1.0"},
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{release.asset_name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary_path.open("wb") as handle:
            with urlopen(request, timeout=timeout_seconds) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Model release redirected to a non-HTTPS URL")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != release.size_bytes:
                    raise ValueError("Model release Content-Length differs from the contract")
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > release.size_bytes:
                        raise ValueError("Model release exceeds the configured byte count")
                    digest.update(chunk)
                    handle.write(chunk)
        if total != release.size_bytes:
            raise ValueError("Model release byte count differs from the contract")
        actual_hash = digest.hexdigest()
        if actual_hash != release.sha256:
            raise ValueError("Model release SHA-256 differs from the contract")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return release.sha256
