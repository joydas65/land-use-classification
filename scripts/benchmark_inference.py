"""Benchmark the versioned serving artifact on leakage-controlled test images."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from terraclass.data import file_sha256
from terraclass.inference import TerraClassPredictor, load_serving_config


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "minimum": min(values),
        "maximum": max(values),
    }


def benchmark(
    project_root: Path,
    config_path: Path,
    dataset_root: Path,
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    config = load_serving_config(config_path)
    manifest_path = project_root / config.training_manifest_path
    if file_sha256(manifest_path) != config.training_manifest_sha256:
        raise RuntimeError("Benchmark manifest differs from the serving contract")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        test_rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if not test_rows:
        raise RuntimeError("Serving manifest does not contain test samples")
    requests = [
        {
            "class_name": row["class_name"],
            "payload": (dataset_root / row["relative_path"]).read_bytes(),
        }
        for row in test_rows
    ]

    load_started = time.perf_counter()
    predictor = TerraClassPredictor.load(config, project_root, device=device)
    load_ms = (time.perf_counter() - load_started) * 1000
    for index in range(warmup):
        predictor.predict_bytes(requests[index % len(requests)]["payload"])

    request_latencies: list[float] = []
    model_pipeline_latencies: list[float] = []
    correct = 0
    measured_started = time.perf_counter()
    for index in range(iterations):
        request = requests[index % len(requests)]
        started = time.perf_counter()
        prediction = predictor.predict_bytes(request["payload"])
        request_latencies.append((time.perf_counter() - started) * 1000)
        model_pipeline_latencies.append(prediction.latency_ms)
        correct += prediction.predicted_class == request["class_name"]
    measured_seconds = time.perf_counter() - measured_started

    return {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "model": {
            "model_id": config.model_id,
            "model_version": config.model_version,
            "architecture": config.architecture,
            "serving_artifact_sha256": config.serving_artifact.sha256,
            "training_manifest_sha256": config.training_manifest_sha256,
        },
        "environment": {
            "device": str(predictor.device),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "torch_threads": torch.get_num_threads(),
        },
        "protocol": {
            "warmup_requests": warmup,
            "measured_requests": iterations,
            "unique_test_images": len(test_rows),
            "input_mode": "in-memory encoded image bytes",
            "request_latency_scope": "decode, validation, preprocessing, and model inference",
            "model_pipeline_scope": "preprocessing and model inference",
        },
        "model_load_ms": load_ms,
        "request_latency_ms": summarize(request_latencies),
        "model_pipeline_latency_ms": summarize(model_pipeline_latencies),
        "throughput_requests_per_second": iterations / measured_seconds,
        "prediction_accuracy_sanity_check": correct / iterations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/serving/resnet18_group_aware_v1.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/UCMerced_LandUse/Images"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/inference_benchmark_2026-07-15.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    dataset_root = (
        args.dataset_root if args.dataset_root.is_absolute() else project_root / args.dataset_root
    )
    report = benchmark(
        project_root,
        config_path,
        dataset_root,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
