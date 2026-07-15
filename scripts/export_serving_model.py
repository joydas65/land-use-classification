"""Promote a verified training checkpoint to a minimal weights-only serving artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def promote_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
    model_id: str,
    model_version: str,
) -> str:
    """Verify, minimize, atomically write, and safely reload a serving artifact."""
    if sha256(source) != expected_source_sha256:
        raise ValueError("Training checkpoint SHA-256 differs from the expected value")
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "model_state_dict",
        "class_names",
        "baseline_config",
        "transfer_config",
        "manifest_sha256",
        "selected",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Training checkpoint is missing fields: {sorted(missing)}")
    transfer_config = checkpoint["transfer_config"]
    artifact = {
        "schema_version": 1,
        "model_id": model_id,
        "model_version": model_version,
        "architecture": str(transfer_config["architecture"]),
        "class_names": list(checkpoint["class_names"]),
        "dropout": float(transfer_config["dropout"]),
        "manifest_sha256": str(checkpoint["manifest_sha256"]),
        "selected_epoch": int(checkpoint["selected"]["global_epoch"]),
        "source_checkpoint_sha256": expected_source_sha256,
        "preprocessing": _json_safe(checkpoint["baseline_config"]["preprocessing"]),
        "model_state_dict": checkpoint["model_state_dict"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(artifact, temporary_path)
        reloaded = torch.load(temporary_path, map_location="cpu", weights_only=True)
        if set(reloaded) != set(artifact):
            raise RuntimeError("Serving artifact changed during safe reload")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sha256(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_sha256 = promote_checkpoint(
        args.source,
        args.output,
        expected_source_sha256=args.expected_source_sha256,
        model_id=args.model_id,
        model_version=args.model_version,
    )
    print(
        json.dumps(
            {
                "source": str(args.source),
                "source_sha256": args.expected_source_sha256,
                "serving_artifact": str(args.output),
                "serving_artifact_sha256": artifact_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
