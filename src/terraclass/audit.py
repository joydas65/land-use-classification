"""Cross-check the original notebook, configuration, documentation, and tests."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from terraclass.colab_results import BundleValidationError, audit_versioned_evidence
from terraclass.config import load_config
from terraclass.inference import ServingConfig, load_serving_config
from terraclass.transfer_config import load_transfer_config

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
    group_audit_path = root / "data/GROUP_AWARE_AUDIT.json"
    iit_criteria_path = root / "docs/IIT_SUBMISSION_CRITERIA.md"
    resume_evidence_path = root / "docs/RESUME_EVIDENCE.md"
    transfer_results_document_path = root / "docs/TRANSFER_RESULTS.md"
    transfer_results_report_path = root / "reports/transfer_learning_results_2026-07-12.json"
    submission_notebook_path = root / "notebooks/Improved_Land_Use_Classification_IITK.ipynb"
    colab_handoff_path = root / "docs/COLAB_HANDOFF.md"
    iit_checklist_path = root / "docs/IIT_SUBMISSION_CHECKLIST.md"
    colab_report_dir = root / "reports/colab"
    colab_figure_path = root / "reports/figures/training_and_confusion_colab_l4.png"
    serving_config_path = root / "configs/serving/resnet18_group_aware_v1.json"
    inference_benchmark_path = root / "reports/inference_benchmark_2026-07-15.json"
    inference_document_path = root / "docs/INFERENCE_FOUNDATION.md"
    api_document_path = root / "docs/API_AND_WEB_APP.md"
    api_source_path = root / "src/terraclass/api.py"
    api_test_path = root / "tests/test_api.py"
    pyproject_path = root / "pyproject.toml"
    web_app_path = root / "web/app/TerraClassApp.tsx"
    web_package_path = root / "web/package.json"
    web_test_path = root / "web/tests/rendered-html.test.mjs"
    web_hosting_path = root / "web/.openai/hosting.json"

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

    if not group_audit_path.is_file():
        report.errors.append("data/GROUP_AWARE_AUDIT.json is missing")
        group_audit: dict[str, Any] = {}
    else:
        group_audit = json.loads(group_audit_path.read_text(encoding="utf-8"))
        group_manifest_path = root / group_audit["manifest_path"]
        if _sha256(group_manifest_path) != group_audit["manifest_sha256"]:
            report.errors.append("Group-aware manifest checksum differs from its audit")
        if group_audit.get("split_counts") != config.split.expected_counts:
            report.errors.append("Group-aware split counts differ from baseline split counts")
        if group_audit.get("group_aware_crossings") != {}:
            report.errors.append("Group-aware manifest contains a reviewed group crossing")
        with group_manifest_path.open(encoding="utf-8", newline="") as handle:
            group_rows = list(csv.DictReader(handle))
        group_splits: dict[str, set[str]] = {}
        for row in group_rows:
            group_splits.setdefault(row["group_id"], set()).add(row["split"])
        if any(len(split_names) > 1 for split_names in group_splits.values()):
            report.errors.append("A group ID crosses boundaries in the group-aware manifest")

    transfer_config_paths = sorted((root / "configs/transfer").glob("*.json"))
    if len(transfer_config_paths) != 4:
        report.errors.append("Exactly four transfer-learning configurations are required")
    transfer_matrix: set[tuple[str, str]] = set()
    for transfer_config_path in transfer_config_paths:
        transfer_config = load_transfer_config(transfer_config_path)
        transfer_matrix.add((transfer_config.architecture, transfer_config.split_kind))
        transfer_manifest = root / transfer_config.manifest_path
        if not transfer_manifest.is_file():
            report.errors.append(f"Transfer manifest is missing: {transfer_config.manifest_path}")
        elif _sha256(transfer_manifest) != transfer_config.manifest_sha256:
            report.errors.append(
                f"Transfer manifest hash differs for {transfer_config.experiment_name}"
            )
    expected_transfer_matrix = {
        (architecture, split_kind)
        for architecture in ("resnet18", "efficientnet_b0")
        for split_kind in ("historical", "group_aware")
    }
    if transfer_matrix != expected_transfer_matrix:
        report.errors.append("Transfer configs do not cover both architectures and split tracks")

    if not transfer_results_report_path.is_file():
        report.errors.append("Versioned transfer-learning results report is missing")
        transfer_results: dict[str, Any] = {}
    else:
        transfer_results = json.loads(transfer_results_report_path.read_text(encoding="utf-8"))
        completed_runs = transfer_results.get("completed_runs", [])
        if len(completed_runs) != 2:
            report.errors.append("Transfer report must contain two completed ResNet18 runs")
        completed_splits = {run.get("split_kind") for run in completed_runs}
        if completed_splits != {"historical", "group_aware"}:
            report.errors.append("Transfer report lacks both completed split tracks")
        expected_manifest_hashes = {
            "historical": config.split.manifest_sha256,
            "group_aware": group_audit.get("manifest_sha256"),
        }
        for run in completed_runs:
            split_kind = run.get("split_kind")
            if run.get("architecture") != "resnet18":
                report.errors.append("A completed selected-model run is not ResNet18")
            if run.get("manifest_sha256") != expected_manifest_hashes.get(split_kind):
                report.errors.append(f"Transfer result manifest differs for {split_kind}")
            for metric in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy"):
                if run.get("test", {}).get(metric) != 1.0:
                    report.errors.append(f"Transfer result {split_kind} {metric} differs from 1.0")
    if any(
        run.get("result_claim_allowed") is not False
        for run in transfer_results.get("incomplete_runs", [])
    ):
        report.errors.append("An incomplete run is incorrectly marked as claimable")

    try:
        colab_verification = audit_versioned_evidence(
            colab_report_dir,
            colab_figure_path,
            manifest_path,
            root / group_audit.get("manifest_path", "missing-group-manifest"),
        )
    except (BundleValidationError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        report.errors.append(f"Versioned Colab evidence failed validation: {error}")
        colab_verification: dict[str, Any] = {}

    if not submission_notebook_path.is_file():
        report.errors.append("Self-contained IIT submission notebook is missing")
        submission_notebook: dict[str, Any] = {"cells": [], "metadata": {}}
    else:
        submission_notebook = json.loads(submission_notebook_path.read_text(encoding="utf-8"))
        if submission_notebook.get("nbformat") != 4:
            report.errors.append("Submission notebook is not nbformat 4")
        if submission_notebook.get("metadata", {}).get("accelerator") != "GPU":
            report.errors.append("Submission notebook does not request a GPU accelerator")
        submission_source = "\n".join(
            str(cell.get("source", "")) for cell in submission_notebook.get("cells", [])
        )
        submission_tokens = (
            config.split.manifest_sha256,
            group_audit.get("manifest_sha256", "missing-group-manifest-hash"),
            config.dataset.archive_sha256,
            "ResNet18",
            "EfficientNet-B0",
            "validation macro F1",
            "classification_report",
            "NVIDIA L4",
            "Selected final architecture: ResNet18",
        )
        for token in submission_tokens:
            if token not in submission_source:
                report.errors.append(f"Submission notebook token is missing: {token}")
        for forbidden in (
            "/Users/",
            "kaggle.json",
            "KAGGLE_KEY",
            "GITHUB_TOKEN",
            "ghp_",
            "TO_BE_FILLED",
            "2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae",
            "414233c8471ea961bfd9406a33f54b427e75ab49",
        ):
            if forbidden in submission_source:
                report.errors.append(f"Submission notebook contains forbidden token: {forbidden}")
        attachment_count = sum(
            len(cell.get("attachments", {})) for cell in submission_notebook.get("cells", [])
        )
        if attachment_count:
            report.errors.append("Submission notebook contains a Colab-incompatible attachment")
        verified_output_cells = [
            cell
            for cell in submission_notebook.get("cells", [])
            if "verified-gpu-output" in cell.get("metadata", {}).get("tags", [])
        ]
        if len(verified_output_cells) != 1:
            report.errors.append("Submission notebook must contain one verified GPU image output")
        else:
            verified_output_cell = verified_output_cells[0]
            verified_outputs = verified_output_cell.get("outputs", [])
            image_payload = (
                verified_outputs[0].get("data", {}).get("image/png")
                if len(verified_outputs) == 1
                and verified_outputs[0].get("output_type") == "display_data"
                else None
            )
            if not image_payload:
                report.errors.append("Verified GPU output is not a standard PNG display output")
            elif hashlib.sha256(base64.b64decode(image_payload)).hexdigest() != (
                colab_verification.get("figure", {}).get("sha256")
            ):
                report.errors.append("Submission notebook output differs from verified GPU figure")
        for index, cell in enumerate(submission_notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            is_verified_output = "verified-gpu-output" in cell.get("metadata", {}).get("tags", [])
            if is_verified_output:
                if cell.get("execution_count") != 1:
                    report.errors.append(
                        "Submission notebook verified output cell "
                        f"{index} has invalid execution count"
                    )
            elif cell.get("outputs") or cell.get("execution_count") is not None:
                report.errors.append(
                    f"Submission notebook cell {index} contains saved execution state"
                )
            python_source = "\n".join(
                line
                for line in str(cell.get("source", "")).splitlines()
                if not line.lstrip().startswith("%")
            )
            try:
                compile(python_source, f"submission-cell-{index}", "exec")
            except SyntaxError as error:
                report.errors.append(f"Submission notebook cell {index} does not compile: {error}")

    serving_config: ServingConfig | None = None
    serving_source_hash: str | None = None
    serving_artifact_hash: str | None = None
    if not serving_config_path.is_file():
        report.errors.append("Versioned serving configuration is missing")
    else:
        try:
            serving_config = load_serving_config(serving_config_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            report.errors.append(f"Serving configuration is invalid: {error}")
        if serving_config is not None:
            if serving_config.architecture != "resnet18":
                report.errors.append("Serving configuration does not select ResNet18")
            if serving_config.class_names != config.dataset.selected_classes:
                report.errors.append("Serving class order differs from the experiment config")
            if serving_config.training_manifest_sha256 != group_audit.get("manifest_sha256"):
                report.errors.append("Serving manifest differs from the group-aware audit")
            if serving_config.selected_epoch != 4:
                report.errors.append("Serving configuration does not select verified epoch 4")
            if serving_config.test_accuracy != 1.0 or serving_config.test_macro_f1 != 1.0:
                report.errors.append("Serving metrics differ from the verified group-aware result")
            source_checkpoint_path = root / serving_config.source_checkpoint.path
            if source_checkpoint_path.is_file():
                serving_source_hash = _sha256(source_checkpoint_path)
                if serving_source_hash != serving_config.source_checkpoint.sha256:
                    report.errors.append("Local training checkpoint differs from serving config")
            serving_artifact_path = root / serving_config.serving_artifact.path
            if serving_artifact_path.is_file():
                serving_artifact_hash = _sha256(serving_artifact_path)
                if serving_artifact_hash != serving_config.serving_artifact.sha256:
                    report.errors.append("Local serving artifact differs from serving config")

    if not inference_benchmark_path.is_file():
        report.errors.append("Versioned inference benchmark is missing")
        inference_benchmark: dict[str, Any] = {}
    else:
        inference_benchmark = json.loads(inference_benchmark_path.read_text(encoding="utf-8"))
        if serving_config is not None:
            benchmark_model = inference_benchmark.get("model", {})
            expected_benchmark_model = {
                "model_id": serving_config.model_id,
                "model_version": serving_config.model_version,
                "architecture": serving_config.architecture,
                "serving_artifact_sha256": serving_config.serving_artifact.sha256,
                "training_manifest_sha256": serving_config.training_manifest_sha256,
            }
            if benchmark_model != expected_benchmark_model:
                report.errors.append("Inference benchmark model identity differs from config")
        benchmark_protocol = inference_benchmark.get("protocol", {})
        if benchmark_protocol.get("measured_requests") != 75:
            report.errors.append("Inference benchmark must contain 75 measured requests")
        if benchmark_protocol.get("unique_test_images") != 75:
            report.errors.append("Inference benchmark must cover all 75 group-aware test images")
        if inference_benchmark.get("prediction_accuracy_sanity_check") != 1.0:
            report.errors.append("Inference benchmark prediction sanity check differs from 1.0")
        request_latency = inference_benchmark.get("request_latency_ms", {})
        p50 = request_latency.get("p50")
        p95 = request_latency.get("p95")
        if not isinstance(p50, (int, float)) or not isinstance(p95, (int, float)):
            report.errors.append("Inference benchmark is missing numeric p50/p95 latency")
        elif not 0 < p50 <= p95:
            report.errors.append("Inference benchmark p50/p95 latency is inconsistent")
        if inference_benchmark.get("throughput_requests_per_second", 0) <= 0:
            report.errors.append("Inference benchmark throughput must be positive")

    if not api_source_path.is_file():
        report.errors.append("Typed inference API source is missing")
        api_source = ""
    else:
        api_source = api_source_path.read_text(encoding="utf-8")
    expected_api_routes = (
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/api/v1/model",
        "/api/v1/predictions",
    )
    for route in expected_api_routes:
        if route not in api_source:
            report.errors.append(f"Versioned API route is missing: {route}")
    for token in (
        "CORSMiddleware",
        "X-Request-ID",
        "RequestValidationError",
        "run_in_threadpool",
        "max_image_bytes",
    ):
        if token not in api_source:
            report.errors.append(f"API safety/operability token is missing: {token}")

    api_test_count = 0
    if not api_test_path.is_file():
        report.errors.append("API contract tests are missing")
    else:
        api_test_source = api_test_path.read_text(encoding="utf-8")
        api_test_count = len(re.findall(r"^def test_", api_test_source, re.MULTILINE))
        if api_test_count != 5:
            report.errors.append("API contract suite must contain five focused tests")

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    web_dependencies = pyproject.get("project", {}).get("optional-dependencies", {}).get("web", [])
    if not any(str(dependency).startswith("fastapi") for dependency in web_dependencies):
        report.errors.append("FastAPI is missing from the web optional dependency set")
    if pyproject.get("project", {}).get("scripts", {}).get("terraclass-api") != (
        "terraclass.api:main"
    ):
        report.errors.append("terraclass-api command does not target terraclass.api:main")

    web_test_count = 0
    if not web_app_path.is_file():
        report.errors.append("TerraClass browser application source is missing")
        web_app_source = ""
    else:
        web_app_source = web_app_path.read_text(encoding="utf-8")
    for token in (
        "NEXT_PUBLIC_TERRACLASS_API_URL",
        "/api/v1/health/ready",
        "/api/v1/predictions?top_k=",
        'aria-live="polite"',
        "not a universal satellite classifier",
    ):
        if token not in web_app_source:
            report.errors.append(f"Browser application contract token is missing: {token}")
    if not web_package_path.is_file():
        report.errors.append("Browser application package manifest is missing")
        web_package: dict[str, Any] = {}
    else:
        web_package = json.loads(web_package_path.read_text(encoding="utf-8"))
        if web_package.get("name") != "terraclass-web":
            report.errors.append("Browser package name is not terraclass-web")
        if web_package.get("scripts", {}).get("test") != (
            "vinext build && node --test tests/rendered-html.test.mjs"
        ):
            report.errors.append("Browser test command does not build before render testing")
        forbidden_web_dependencies = {"react-loading-skeleton", "drizzle-orm", "drizzle-kit"}
        declared_web_dependencies = {
            *web_package.get("dependencies", {}),
            *web_package.get("devDependencies", {}),
        }
        unexpected_web_dependencies = forbidden_web_dependencies & declared_web_dependencies
        if unexpected_web_dependencies:
            report.errors.append(
                "Unused starter dependency remains: "
                + ", ".join(sorted(unexpected_web_dependencies))
            )
    if not web_test_path.is_file():
        report.errors.append("Browser server-render tests are missing")
    else:
        web_test_source = web_test_path.read_text(encoding="utf-8")
        web_test_count = len(re.findall(r'^test\("', web_test_source, re.MULTILINE))
        if web_test_count != 2:
            report.errors.append("Browser suite must contain two focused render tests")
    if not web_hosting_path.is_file():
        report.errors.append("Sites hosting configuration is missing")
    else:
        hosting_config = json.loads(web_hosting_path.read_text(encoding="utf-8"))
        unexpected_hosting_keys = set(hosting_config) - {"project_id", "d1", "r2"}
        if unexpected_hosting_keys:
            report.errors.append("Sites hosting configuration contains unexpected keys")

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
    if not iit_criteria_path.is_file():
        report.errors.append("docs/IIT_SUBMISSION_CRITERIA.md is missing")
        iit_criteria = ""
    else:
        iit_criteria = iit_criteria_path.read_text(encoding="utf-8")
    if not resume_evidence_path.is_file():
        report.errors.append("docs/RESUME_EVIDENCE.md is missing")
        resume_evidence = ""
    else:
        resume_evidence = resume_evidence_path.read_text(encoding="utf-8")
    if not transfer_results_document_path.is_file():
        report.errors.append("docs/TRANSFER_RESULTS.md is missing")
        transfer_results_document = ""
    else:
        transfer_results_document = transfer_results_document_path.read_text(encoding="utf-8")
    if not colab_handoff_path.is_file():
        report.errors.append("docs/COLAB_HANDOFF.md is missing")
        colab_handoff = ""
    else:
        colab_handoff = colab_handoff_path.read_text(encoding="utf-8")
    if not iit_checklist_path.is_file():
        report.errors.append("docs/IIT_SUBMISSION_CHECKLIST.md is missing")
        iit_checklist = ""
    else:
        iit_checklist = iit_checklist_path.read_text(encoding="utf-8")
    if not inference_document_path.is_file():
        report.errors.append("docs/INFERENCE_FOUNDATION.md is missing")
        inference_document = ""
    else:
        inference_document = inference_document_path.read_text(encoding="utf-8")
    if not api_document_path.is_file():
        report.errors.append("docs/API_AND_WEB_APP.md is missing")
        api_document = ""
    else:
        api_document = api_document_path.read_text(encoding="utf-8")
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
    combined_documentation = "\n".join(
        (
            readme,
            audit_document,
            reproduction_document,
            iit_criteria,
            resume_evidence,
            transfer_results_document,
            colab_handoff,
            iit_checklist,
            inference_document,
            api_document,
        )
    )
    for token in required_documentation_tokens:
        if token not in combined_documentation:
            report.errors.append(f"Required baseline documentation token is missing: {token}")
    if serving_config is not None and inference_benchmark:
        inference_documentation_tokens = (
            serving_config.model_id,
            serving_config.source_checkpoint.sha256,
            serving_config.serving_artifact.sha256,
            f"{inference_benchmark['request_latency_ms']['p50']:.1f} ms",
            f"{inference_benchmark['request_latency_ms']['p95']:.1f} ms",
            "weights-only",
        )
        for token in inference_documentation_tokens:
            if token not in inference_document:
                report.errors.append(f"Inference documentation token is missing: {token}")
    for token in (
        "Submitted 15 July 2026",
        "submitted by email on 15 July 2026",
    ):
        if token not in "\n".join((iit_criteria, iit_checklist)):
            report.errors.append(f"IIT submission status token is missing: {token}")
    for token in (
        "FastAPI",
        "request ID",
        "NEXT_PUBLIC_TERRACLASS_API_URL",
        "not yet claimed as a deployed integrated system",
    ):
        if token not in api_document:
            report.errors.append(f"Application documentation token is missing: {token}")

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
        "group_aware_manifest": {
            "sha256": group_audit.get("manifest_sha256"),
            "groups": group_audit.get("grouping", {}).get("total_groups"),
            "reviewed_multi_image_groups": group_audit.get("grouping", {}).get(
                "multi_image_groups"
            ),
            "crossings": group_audit.get("group_aware_crossings"),
        },
        "transfer_experiment_matrix": sorted(
            f"{architecture}:{split_kind}" for architecture, split_kind in transfer_matrix
        ),
        "local_cpu_transfer_runs": [
            {
                "experiment_name": run.get("experiment_name"),
                "split_kind": run.get("split_kind"),
                "test_accuracy": run.get("test", {}).get("accuracy"),
                "test_macro_f1": run.get("test", {}).get("macro_f1"),
            }
            for run in transfer_results.get("completed_runs", [])
        ],
        "verified_colab_gpu": {
            "bundle_sha256": colab_verification.get("source_bundle", {}).get("sha256"),
            "gpu": colab_verification.get("hardware", {}).get("gpu"),
            "completed_matrix": colab_verification.get("completed_matrix"),
            "failures": colab_verification.get("failures"),
            "selected_architecture": colab_verification.get("selected_architecture"),
        },
        "submission_notebook": {
            "cells": len(submission_notebook.get("cells", [])),
            "gpu_requested": submission_notebook.get("metadata", {}).get("accelerator") == "GPU",
            "saved_outputs": sum(
                bool(cell.get("outputs")) for cell in submission_notebook.get("cells", [])
            ),
            "attachments": sum(
                len(cell.get("attachments", {})) for cell in submission_notebook.get("cells", [])
            ),
            "verified_image_outputs": sum(
                "verified-gpu-output" in cell.get("metadata", {}).get("tags", [])
                for cell in submission_notebook.get("cells", [])
            ),
        },
        "serving_foundation": {
            "model_id": serving_config.model_id if serving_config else None,
            "model_version": serving_config.model_version if serving_config else None,
            "training_manifest_sha256": (
                serving_config.training_manifest_sha256 if serving_config else None
            ),
            "source_checkpoint_present": serving_source_hash is not None,
            "source_checkpoint_sha256": serving_source_hash,
            "serving_artifact_present": serving_artifact_hash is not None,
            "serving_artifact_sha256": serving_artifact_hash,
            "request_latency_p50_ms": inference_benchmark.get("request_latency_ms", {}).get("p50"),
            "request_latency_p95_ms": inference_benchmark.get("request_latency_ms", {}).get("p95"),
            "throughput_requests_per_second": inference_benchmark.get(
                "throughput_requests_per_second"
            ),
        },
        "application_layer": {
            "api_routes": list(expected_api_routes),
            "api_contract_tests": api_test_count,
            "web_render_tests": web_test_count,
            "browser_package": web_package.get("name"),
            "deployment_claimed": False,
        },
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
