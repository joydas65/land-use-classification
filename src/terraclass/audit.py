"""Cross-check the original notebook, configuration, documentation, and tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from terraclass.config import load_config

KNOWN_ISSUE_IDS = (
    "ART-001",
    "CODE-001",
    "DATA-001",
    "DATA-002",
    "DOC-001",
    "EVAL-001",
    "LEAK-001",
    "PATH-001",
    "PORT-001",
    "REPRO-001",
    "SEC-001",
    "TEST-001",
)


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joined_source(notebook: dict[str, Any]) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _joined_text_outputs(notebook: dict[str, Any]) -> str:
    values: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text", [])
            values.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(values)


def _assignment(source: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^#\n]+)", source, re.MULTILINE)
    return match.group(1).strip() if match else None


def audit_project(project_root: str | Path) -> AuditReport:
    root = Path(project_root)
    report = AuditReport()
    config = load_config(root / "configs/baseline_5class.json")
    notebook_path = root / "notebooks/original/Land_Use_Classification.ipynb"
    checksum_path = root / "notebooks/original/SHA256SUMS"
    audit_document_path = root / "docs/BASELINE_AUDIT.md"
    reproduction_document_path = root / "docs/REPRODUCTION_RUN.md"
    reproduction_report_path = root / "reports/baseline_reproduction_2026-07-12.json"
    readme_path = root / "README.md"
    manifest_path = root / config.split.manifest_path
    dataset_audit_path = root / "data/DATASET_AUDIT.json"

    expected_checksum = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_checksum = _sha256(notebook_path)
    if actual_checksum != expected_checksum:
        report.errors.append("Original notebook checksum does not match SHA256SUMS")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = _joined_source(notebook)
    outputs = _joined_text_outputs(notebook)
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    output_cells = [cell for cell in code_cells if cell.get("outputs")]

    assignments = {
        "class_limit": _assignment(source, "NUM_CLASSES_TO_USE"),
        "batch_size": _assignment(source, "batch_size"),
        "learning_rate": _assignment(source, "learning_rate"),
        "epochs": _assignment(source, "num_epochs"),
    }
    expected_assignments = {
        "class_limit": str(config.class_count),
        "batch_size": str(config.training.batch_size),
        "learning_rate": str(config.training.learning_rate),
        "epochs": str(config.training.epochs),
    }
    for key, expected in expected_assignments.items():
        if assignments[key] != expected:
            report.errors.append(
                f"Notebook {key}={assignments[key]!r} differs from config value {expected!r}"
            )
    if config.dataset.source_url not in source:
        report.errors.append("Dataset source URL differs between notebook and config")

    if not manifest_path.is_file():
        report.errors.append(f"Configured manifest does not exist: {config.split.manifest_path}")
        manifest_rows: list[dict[str, str]] = []
        manifest_hash = None
    else:
        manifest_hash = _sha256(manifest_path)
        if manifest_hash != config.split.manifest_sha256:
            report.errors.append("Versioned baseline manifest checksum differs from config")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        manifest_counts = {
            split_name: sum(row["split"] == split_name for row in manifest_rows)
            for split_name in ("train", "validation", "test")
        }
        if manifest_counts != config.split.expected_counts:
            report.errors.append(
                f"Manifest split counts {manifest_counts} differ from configured counts"
            )
        relative_paths = [row["relative_path"] for row in manifest_rows]
        if len(relative_paths) != len(set(relative_paths)):
            report.errors.append("Versioned manifest contains duplicate image paths")
        if any(len(row["sha256"]) != 64 for row in manifest_rows):
            report.errors.append("Versioned manifest contains an invalid image SHA-256")

    if not dataset_audit_path.is_file():
        report.errors.append("data/DATASET_AUDIT.json is missing")
        dataset_audit: dict[str, Any] = {}
    else:
        dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
        expected_dataset_observations = {
            "total_images": 2100,
            "total_classes": 21,
            "unique_content_hashes": 2100,
            "exact_duplicate_group_count": 0,
        }
        for key, expected in expected_dataset_observations.items():
            if dataset_audit.get(key) != expected:
                report.errors.append(
                    f"Dataset audit {key}={dataset_audit.get(key)!r} differs from {expected!r}"
                )
        if dataset_audit.get("dimensions", {}).get("256x256") != 2056:
            report.errors.append("Dataset dimension audit no longer reports 2,056 256x256 images")
        if dataset_audit.get("perceptual_hash", {}).get("candidate_pair_count") != 31:
            report.errors.append("Dataset perceptual candidate count differs from audited value 31")
        baseline_split = dataset_audit.get("baseline_split", {})
        if baseline_split.get("perceptual_candidate_pair_count") != 14:
            report.errors.append("Five-class perceptual candidate count differs from 14")
        if baseline_split.get("perceptual_candidate_cross_split_count") != 8:
            report.errors.append("Five-class cross-split perceptual candidate count differs from 8")

    download_metadata_path = root / "data/raw/DOWNLOAD_METADATA.json"
    if download_metadata_path.is_file():
        download_metadata = json.loads(download_metadata_path.read_text(encoding="utf-8"))
        if download_metadata.get("archive_sha256") != config.dataset.archive_sha256:
            report.errors.append("Downloaded archive SHA-256 differs from config")
        if download_metadata.get("archive_size_bytes") != config.dataset.archive_size_bytes:
            report.errors.append("Downloaded archive size differs from config")

    accuracy_match = re.search(r"Test Accuracy:\s*([0-9.]+)%", outputs)
    macro_match = re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+75", outputs)
    parameter_match = re.search(r"Lightweight CNN Parameters:\s*([0-9,]+)", outputs)
    observed_accuracy = float(accuracy_match.group(1)) / 100 if accuracy_match else None
    observed_macro_f1 = float(macro_match.group(1)) if macro_match else None
    observed_parameters = (
        int(parameter_match.group(1).replace(",", "")) if parameter_match else None
    )
    observed_checks = (
        ("test accuracy", observed_accuracy, config.observed_baseline.test_accuracy),
        ("macro F1", observed_macro_f1, config.observed_baseline.test_macro_f1),
        ("parameter count", observed_parameters, config.observed_baseline.parameter_count),
    )
    for name, observed, expected in observed_checks:
        if observed is None or abs(observed - expected) > 1e-6:
            report.errors.append(f"Notebook {name}={observed!r} differs from config={expected!r}")

    audit_document = audit_document_path.read_text(encoding="utf-8")
    reproduction_document = reproduction_document_path.read_text(encoding="utf-8")
    reproduction_report = json.loads(reproduction_report_path.read_text(encoding="utf-8"))
    if reproduction_report["split_manifest"]["sha256"] != config.split.manifest_sha256:
        report.errors.append("Reproduction report manifest hash differs from config")
    if reproduction_report["dataset"]["archive_sha256"] != config.dataset.archive_sha256:
        report.errors.append("Reproduction report archive hash differs from config")
    if reproduction_report["historical_notebook"]["test_accuracy"] != (
        config.observed_baseline.test_accuracy
    ):
        report.errors.append("Reproduction report historical accuracy differs from config")
    reproduction_tokens = (
        f"{reproduction_report['test']['accuracy']:.2%}",
        f"{reproduction_report['test']['macro_f1']:.3f}",
        f"{reproduction_report['training']['best_validation_accuracy']:.2%}",
        reproduction_report["split_manifest"]["sha256"],
    )
    for token in reproduction_tokens:
        if token not in reproduction_document:
            report.errors.append(f"Reproduction documentation token is missing: {token}")
    readme = readme_path.read_text(encoding="utf-8")
    for issue_id in KNOWN_ISSUE_IDS:
        if issue_id not in audit_document:
            report.errors.append(f"Known issue {issue_id} is missing from BASELINE_AUDIT.md")
    required_documentation_tokens = (
        "5 classes",
        "21 classes",
        "500 images",
        "350/75/75",
        "102,277",
        "74.67%",
        "0.733",
    )
    combined_documentation = readme + "\n" + audit_document + "\n" + reproduction_document
    for token in required_documentation_tokens:
        if token not in combined_documentation:
            report.errors.append(f"Required baseline documentation token is missing: {token}")

    report.warnings.extend(
        [
            f"{issue_id}: accepted original-notebook issue; see docs/BASELINE_AUDIT.md"
            for issue_id in KNOWN_ISSUE_IDS
        ]
    )
    report.observed = {
        "original_notebook_sha256": actual_checksum,
        "notebook_cells": len(notebook["cells"]),
        "code_cells": len(code_cells),
        "code_cells_with_outputs": len(output_cells),
        "assignments": assignments,
        "test_accuracy": observed_accuracy,
        "test_macro_f1": observed_macro_f1,
        "parameter_count": observed_parameters,
        "baseline_manifest_sha256": manifest_hash,
        "baseline_manifest_rows": len(manifest_rows),
        "dataset_audit": {
            "images_256x256": dataset_audit.get("dimensions", {}).get("256x256"),
            "exact_duplicate_groups": dataset_audit.get("exact_duplicate_group_count"),
            "perceptual_candidate_pairs": dataset_audit.get("perceptual_hash", {}).get(
                "candidate_pair_count"
            ),
            "five_class_cross_split_candidates": dataset_audit.get("baseline_split", {}).get(
                "perceptual_candidate_cross_split_count"
            ),
        },
        "controlled_reproduction": {
            "test_accuracy": reproduction_report["test"]["accuracy"],
            "test_macro_f1": reproduction_report["test"]["macro_f1"],
            "selected_epoch": reproduction_report["training"]["selected_epoch"],
        },
        "known_issue_ids": list(KNOWN_ISSUE_IDS),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_project(args.project_root)
    if args.json:
        print(json.dumps({**asdict(report), "ok": report.ok}, indent=2))
    else:
        print(f"Audit status: {'PASS' if report.ok else 'FAIL'}")
        for error in report.errors:
            print(f"ERROR: {error}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        print(json.dumps(report.observed, indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
