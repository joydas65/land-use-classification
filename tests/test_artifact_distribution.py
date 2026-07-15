import hashlib
import io
import json
from pathlib import Path

import pytest

import terraclass.artifact_distribution as distribution
from terraclass.artifact_distribution import (
    ModelRelease,
    fetch_release_artifact,
    load_model_release,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, url: str = "https://assets.example/model.pt") -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _release(payload: bytes, **overrides) -> ModelRelease:
    values = {
        "schema_version": 1,
        "model_id": "terraclass-resnet18-group-aware",
        "model_version": "1.0.0",
        "asset_name": "model.pt",
        "url": "https://example.test/model.pt",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    values.update(overrides)
    return ModelRelease(**values)


def test_model_release_matches_the_serving_contract(project_root: Path) -> None:
    release = load_model_release(project_root / "configs/serving/model_release_v1.json")
    serving = json.loads(
        (project_root / "configs/serving/resnet18_group_aware_v1.json").read_text(encoding="utf-8")
    )
    assert release.model_id == serving["model_id"]
    assert release.model_version == serving["model_version"]
    assert release.sha256 == serving["serving_artifact"]["sha256"]
    assert release.size_bytes == 44_795_275


def test_fetch_release_artifact_is_atomic_and_checksum_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-model-bytes"
    release = _release(payload)
    monkeypatch.setattr(
        distribution,
        "urlopen",
        lambda request, timeout: FakeResponse(payload),
    )
    destination = tmp_path / "model.pt"
    assert fetch_release_artifact(release, destination) == release.sha256
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_fetch_rejects_wrong_bytes_without_replacing_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected-model"
    received = b"tampered-model"
    release = _release(expected, size_bytes=len(received))
    monkeypatch.setattr(
        distribution,
        "urlopen",
        lambda request, timeout: FakeResponse(received),
    )
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"known-good-existing-artifact")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_release_artifact(release, destination)
    assert destination.read_bytes() == b"known-good-existing-artifact"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("url", "http://example.test/model.pt", "HTTPS"),
        ("asset_name", "../model.pt", "plain file name"),
        ("sha256", "not-a-hash", "lowercase SHA-256"),
        ("size_bytes", 0, "positive"),
    ],
)
def test_model_release_validation_rejects_unsafe_values(
    field: str, value: object, message: str
) -> None:
    payload = b"model"
    release = _release(payload, **{field: value})
    with pytest.raises(ValueError, match=message):
        release.validate()
