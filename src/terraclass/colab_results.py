"""Strict validation for the Colab GPU collaboration results bundle."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_FILES = {
    "colab_run_report.json",
    "efficientnet_b0_group_aware.json",
    "efficientnet_b0_historical.json",
    "group_aware_manifest.csv",
    "historical_manifest.csv",
    "model_comparison.csv",
    "resnet18_group_aware.json",
    "resnet18_historical.json",
    "training_and_confusion.png",
}

EXPECTED_CLASSES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
]
EXPECTED_MANIFEST_HASHES = {
    "historical": "73d19e048e742fdf616cbbc1f037efa009ea329ec600acef329f2a5bc7df87ea",
    "group_aware": "26bc3503f6a16e841286771b727e1f1f14a58c623deafe26c45e52d68b88081d",
}
EXPECTED_PARAMETERS = {"resnet18": 11_179_077, "efficientnet_b0": 4_013_953}
EXPECTED_MATRIX = {
    (architecture, split_kind)
    for architecture in ("resnet18", "efficientnet_b0")
    for split_kind in ("historical", "group_aware")
}


class BundleValidationError(ValueError):
    """Raised when the returned Colab evidence violates the experiment contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleValidationError(message)


def _close(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def _png_dimensions(content: bytes) -> tuple[int, int]:
    _require(content.startswith(b"\x89PNG\r\n\x1a\n"), "Figure is not a PNG")
    _require(len(content) >= 24 and content[12:16] == b"IHDR", "PNG lacks an IHDR chunk")
    return struct.unpack(">II", content[16:24])


def _validate_manifest(content: bytes, split_kind: str) -> dict[str, Any]:
    digest = sha256_bytes(content)
    _require(
        digest == EXPECTED_MANIFEST_HASHES[split_kind],
        f"{split_kind} manifest SHA-256 mismatch",
    )
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline="")))
    _require(len(rows) == 500, f"{split_kind} manifest must contain 500 rows")
    counts = {
        split_name: sum(row["split"] == split_name for row in rows)
        for split_name in ("train", "validation", "test")
    }
    _require(counts == {"train": 350, "validation": 75, "test": 75}, "Split mismatch")
    _require(
        len({row["relative_path"] for row in rows}) == 500,
        f"{split_kind} manifest contains duplicate paths",
    )
    _require(
        {row["class_name"] for row in rows} == set(EXPECTED_CLASSES),
        f"{split_kind} manifest class set mismatch",
    )
    group_crossings: dict[str, set[str]] = {}
    if split_kind == "group_aware":
        _require("group_id" in rows[0], "Group-aware manifest lacks group_id")
        for row in rows:
            group_crossings.setdefault(row["group_id"], set()).add(row["split"])
        _require(
            all(len(names) == 1 for names in group_crossings.values()),
            "A group-aware group crosses split boundaries",
        )
    return {
        "sha256": digest,
        "rows": len(rows),
        "split_counts": counts,
        "group_crossings": sum(len(names) > 1 for names in group_crossings.values()),
    }


def _confusion_metrics(matrix: list[list[int]]) -> tuple[float, float]:
    _require(len(matrix) == 5, "Confusion matrix must have five rows")
    _require(all(len(row) == 5 for row in matrix), "Confusion matrix must have five columns")
    _require(
        all(isinstance(value, int) and value >= 0 for row in matrix for value in row),
        "Confusion matrix values must be non-negative integers",
    )
    _require([sum(row) for row in matrix] == [15] * 5, "Each test class must have support 15")
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[index][index] for index in range(5)) / total
    recalls = [matrix[index][index] / sum(matrix[index]) for index in range(5)]
    return accuracy, sum(recalls) / len(recalls)


def _validate_run(run: dict[str, Any]) -> dict[str, Any]:
    architecture = run.get("architecture")
    split_kind = run.get("split_kind")
    key = (architecture, split_kind)
    _require(key in EXPECTED_MATRIX, f"Unexpected experiment matrix entry: {key}")
    _require(run.get("seed") == 42, f"{key} seed differs from 42")
    _require(run.get("device") == "cuda", f"{key} did not run on CUDA")
    _require(
        run.get("manifest_sha256") == EXPECTED_MANIFEST_HASHES[split_kind],
        f"{key} manifest hash mismatch",
    )
    _require(
        run.get("total_parameters") == EXPECTED_PARAMETERS[architecture],
        f"{key} parameter count mismatch",
    )
    _require(float(run.get("elapsed_seconds", 0)) > 0, f"{key} elapsed time must be positive")

    history = run.get("history", [])
    _require(history, f"{key} history is empty")
    global_epochs = [entry.get("global_epoch") for entry in history]
    _require(global_epochs == list(range(1, len(history) + 1)), f"{key} epoch sequence mismatch")
    selected = run.get("selected", {})
    selected_epoch = selected.get("global_epoch")
    _require(selected_epoch in global_epochs, f"{key} selected epoch is absent from history")
    validation_scores = [float(entry["validation"]["macro_f1"]) for entry in history]
    _require(
        _close(selected.get("validation_macro_f1"), max(validation_scores)),
        f"{key} did not select the maximum validation macro F1",
    )
    selected_entry = history[global_epochs.index(selected_epoch)]
    _require(
        selected.get("stage") == selected_entry.get("stage")
        and selected.get("stage_epoch") == selected_entry.get("stage_epoch"),
        f"{key} selected checkpoint metadata mismatch",
    )

    test = run.get("test", {})
    for metric in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy"):
        _require(0 <= float(test.get(metric, -1)) <= 1, f"{key} invalid {metric}")
    confusion_accuracy, confusion_balanced = _confusion_metrics(test["confusion_matrix"])
    _require(_close(test["accuracy"], confusion_accuracy), f"{key} accuracy conflicts with matrix")
    _require(
        _close(test["balanced_accuracy"], confusion_balanced),
        f"{key} balanced accuracy conflicts with matrix",
    )
    report = test.get("classification_report", {})
    for class_name in EXPECTED_CLASSES:
        _require(float(report[class_name]["support"]) == 15, f"{key} class support mismatch")
    _require(float(report["macro avg"]["support"]) == 75, f"{key} total support mismatch")
    return {
        "experiment_name": run["experiment_name"],
        "architecture": architecture,
        "split_kind": split_kind,
        "manifest_sha256": run["manifest_sha256"],
        "total_parameters": run["total_parameters"],
        "selected": selected,
        "elapsed_seconds": run["elapsed_seconds"],
        "test": {
            metric: test[metric]
            for metric in (
                "loss",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "top3_accuracy",
                "confusion_matrix",
            )
        },
    }


def _validate_comparison(content: bytes, summaries: list[dict[str, Any]]) -> None:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    _require(len(rows) == 5, "Comparison CSV must contain the baseline and four GPU runs")
    baseline = rows[0]
    _require(baseline["model"] == "supplied_custom_cnn", "Comparison baseline model mismatch")
    _require(_close(float(baseline["accuracy"]), 0.7467), "Comparison baseline accuracy mismatch")
    _require(_close(float(baseline["macro_f1"]), 0.733), "Comparison baseline F1 mismatch")
    indexed = {(summary["architecture"], summary["split_kind"]): summary for summary in summaries}
    for row in rows[1:]:
        key = (row["model"], row["split"])
        _require(key in indexed, f"Comparison CSV has an unexpected row: {key}")
        summary = indexed[key]
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy"):
            _require(
                _close(float(row[metric]), float(summary["test"][metric])),
                f"Comparison CSV {key} {metric} mismatch",
            )
        _require(
            int(row["selected_epoch"]) == summary["selected"]["global_epoch"],
            f"Comparison CSV {key} selected epoch mismatch",
        )
        _require(
            _close(float(row["elapsed_seconds"]), float(summary["elapsed_seconds"])),
            f"Comparison CSV {key} elapsed time mismatch",
        )


def audit_bundle(
    path: str | Path, *, verified_date: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return canonical verification metadata and validated files from a results ZIP."""
    bundle_path = Path(path)
    _require(bundle_path.is_file(), f"Bundle does not exist: {bundle_path}")
    bundle_sha256 = sha256_file(bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        infos = archive.infolist()
        _require(
            not any(info.flag_bits & 0x1 for info in infos), "Encrypted ZIP entries are forbidden"
        )
        names = [info.filename for info in infos]
        for name in names:
            pure = PurePosixPath(name)
            _require(
                not pure.is_absolute() and ".." not in pure.parts and len(pure.parts) == 1,
                f"Unsafe or nested ZIP path: {name}",
            )
        _require(set(names) == EXPECTED_FILES, "ZIP contents differ from the expected evidence set")
        _require(len(names) == len(set(names)), "ZIP contains duplicate member names")
        _require(sum(info.file_size for info in infos) < 10_000_000, "ZIP expands beyond 10 MB")
        files = {name: archive.read(name) for name in names}

    report = json.loads(files["colab_run_report.json"])
    _require(report.get("schema_version") == 1, "Unsupported Colab report schema")
    _require(report.get("hardware", {}).get("device") == "cuda", "Report is not a CUDA run")
    _require(bool(report.get("hardware", {}).get("gpu")), "GPU model is missing")
    _require(
        report.get("dataset", {}).get("selected_classes") == EXPECTED_CLASSES, "Class mismatch"
    )
    _require(report.get("dataset", {}).get("selected_images") == 500, "Image count mismatch")
    _require(
        report.get("dataset", {}).get("archive_sha256")
        == "06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb",
        "Dataset archive hash mismatch",
    )
    _require(report.get("failures") == [], "Colab report contains failed experiments")
    _require(len(report.get("results", [])) == 4, "Colab report must contain four results")

    manifest_verification = {
        "historical": _validate_manifest(files["historical_manifest.csv"], "historical"),
        "group_aware": _validate_manifest(files["group_aware_manifest.csv"], "group_aware"),
    }
    summaries = []
    observed_matrix = set()
    for run in report["results"]:
        summary = _validate_run(run)
        key = (summary["architecture"], summary["split_kind"])
        observed_matrix.add(key)
        individual_name = f"{summary['experiment_name']}.json"
        _require(individual_name in files, f"Individual result is missing: {individual_name}")
        individual = json.loads(files[individual_name])
        _require(individual == run, f"Individual result differs from report: {individual_name}")
        summaries.append(summary)
    _require(observed_matrix == EXPECTED_MATRIX, "Completed experiment matrix is incomplete")
    _validate_comparison(files["model_comparison.csv"], summaries)
    png_width, png_height = _png_dimensions(files["training_and_confusion.png"])
    _require((png_width, png_height) == (2017, 2538), "Unexpected evidence figure dimensions")

    verification = {
        "schema_version": 1,
        "verified_date": verified_date,
        "source_bundle": {
            "filename": bundle_path.name,
            "sha256": bundle_sha256,
            "size_bytes": bundle_path.stat().st_size,
            "member_count": len(files),
        },
        "hardware": report["hardware"],
        "environment": report["environment"],
        "dataset": report["dataset"],
        "manifests": manifest_verification,
        "failures": report["failures"],
        "completed_matrix": sorted(
            f"{architecture}:{split_kind}" for architecture, split_kind in observed_matrix
        ),
        "runs": summaries,
        "figure": {
            "sha256": sha256_bytes(files["training_and_confusion.png"]),
            "width": png_width,
            "height": png_height,
        },
        "file_sha256": {name: sha256_bytes(content) for name, content in sorted(files.items())},
        "selected_architecture": "resnet18",
        "selection_rationale": (
            "Both architectures tied on classification metrics. ResNet18 had lower test loss on "
            "both manifests, completed the historical L4 run faster, and already has matching "
            "local CPU evidence. EfficientNet-B0 remains the parameter-efficient alternative."
        ),
    }
    return verification, files


def audit_versioned_evidence(
    report_dir: str | Path,
    figure_path: str | Path,
    historical_manifest_path: str | Path,
    group_manifest_path: str | Path,
) -> dict[str, Any]:
    """Revalidate the evidence subset committed to the repository."""
    root = Path(report_dir)
    verification = json.loads((root / "VERIFICATION.json").read_text(encoding="utf-8"))
    _require(verification.get("schema_version") == 1, "Invalid verification schema")
    report_bytes = (root / "colab_run_report.json").read_bytes()
    comparison_bytes = (root / "model_comparison.csv").read_bytes()
    figure_bytes = Path(figure_path).read_bytes()
    expected_file_hashes = verification.get("file_sha256", {})
    _require(
        sha256_bytes(report_bytes) == expected_file_hashes.get("colab_run_report.json"),
        "Versioned Colab report hash mismatch",
    )
    _require(
        sha256_bytes(comparison_bytes) == expected_file_hashes.get("model_comparison.csv"),
        "Versioned model comparison hash mismatch",
    )
    _require(
        sha256_bytes(figure_bytes) == verification.get("figure", {}).get("sha256"),
        "Versioned evidence figure hash mismatch",
    )
    _require(
        _png_dimensions(figure_bytes)
        == (
            verification["figure"]["width"],
            verification["figure"]["height"],
        ),
        "Versioned evidence figure dimensions mismatch",
    )
    committed_manifest_verification = {
        "historical": _validate_manifest(Path(historical_manifest_path).read_bytes(), "historical"),
        "group_aware": _validate_manifest(Path(group_manifest_path).read_bytes(), "group_aware"),
    }
    _require(
        committed_manifest_verification == verification.get("manifests"),
        "Versioned manifest verification differs from committed manifests",
    )

    report = json.loads(report_bytes)
    _require(report.get("failures") == verification.get("failures") == [], "Failure mismatch")
    _require(report.get("hardware") == verification.get("hardware"), "Hardware metadata mismatch")
    _require(
        report.get("environment") == verification.get("environment"),
        "Environment metadata mismatch",
    )
    summaries = [_validate_run(run) for run in report.get("results", [])]
    _require(summaries == verification.get("runs"), "Canonical run summaries have changed")
    _require(
        {(run["architecture"], run["split_kind"]) for run in summaries} == EXPECTED_MATRIX,
        "Versioned experiment matrix is incomplete",
    )
    _validate_comparison(comparison_bytes, summaries)
    _require(
        verification.get("source_bundle", {}).get("sha256")
        == "2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae",
        "Source bundle provenance hash mismatch",
    )
    _require(
        verification.get("selected_architecture") == "resnet18",
        "Selected architecture differs from the reviewed decision",
    )
    return verification
