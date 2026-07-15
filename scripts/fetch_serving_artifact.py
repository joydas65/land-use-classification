"""Fetch the published serving artifact only after release-contract verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terraclass.artifact_distribution import fetch_release_artifact, load_model_release
from terraclass.inference import load_serving_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--serving-config",
        type=Path,
        default=Path("configs/serving/resnet18_group_aware_v1.json"),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=Path("configs/serving/model_release_v1.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    serving_path = (
        args.serving_config if args.serving_config.is_absolute() else root / args.serving_config
    )
    release_path = (
        args.release_config if args.release_config.is_absolute() else root / args.release_config
    )
    serving = load_serving_config(serving_path)
    release = load_model_release(release_path)
    if (release.model_id, release.model_version, release.sha256) != (
        serving.model_id,
        serving.model_version,
        serving.serving_artifact.sha256,
    ):
        raise ValueError("Model release identity differs from the serving configuration")
    destination = root / serving.serving_artifact.path
    artifact_hash = fetch_release_artifact(
        release,
        destination,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "model_id": release.model_id,
                "model_version": release.model_version,
                "destination": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": artifact_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
