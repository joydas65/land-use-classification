from pathlib import Path

import pytest

from scripts.export_serving_model import promote_checkpoint


def test_promotion_rejects_unverified_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="SHA-256"):
        promote_checkpoint(
            source,
            tmp_path / "serving.pt",
            expected_source_sha256="0" * 64,
            model_id="test-model",
            model_version="1.0.0",
        )
