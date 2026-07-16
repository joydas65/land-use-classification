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

from terraclass.artifact_distribution import ModelRelease, load_model_release
from terraclass.colab_results import BundleValidationError, audit_versioned_evidence
from terraclass.config import load_config
from terraclass.inference import ServingConfig, load_serving_config
from terraclass.telemetry import PREDICTION_OBSERVATION_FIELDS, PROHIBITED_PREDICTION_FIELDS
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
    model_release_path = root / "configs/serving/model_release_v1.json"
    model_release_verification_path = root / "reports/model_release_verification_2026-07-16.json"
    container_release_verification_path = (
        root / "reports/container_release_verification_2026-07-16.json"
    )
    inference_benchmark_path = root / "reports/inference_benchmark_2026-07-15.json"
    api_load_report_path = root / "reports/api_load_test_2026-07-16.json"
    cloud_run_load_report_path = root / "reports/cloud_run_load_test_2026-07-16.json"
    cloud_run_deployment_verification_path = (
        root / "reports/cloud_run_deployment_verification_2026-07-16.json"
    )
    cloud_run_scale_to_zero_path = root / "reports/cloud_run_scale_to_zero_2026-07-17.json"
    cloud_monitoring_deployment_path = root / "reports/cloud_monitoring_deployment_2026-07-17.json"
    inference_document_path = root / "docs/INFERENCE_FOUNDATION.md"
    api_document_path = root / "docs/API_AND_WEB_APP.md"
    production_document_path = root / "docs/PRODUCTION_INFERENCE.md"
    observability_document_path = root / "docs/OBSERVABILITY_AND_DRIFT.md"
    api_source_path = root / "src/terraclass/api.py"
    api_test_path = root / "tests/test_api.py"
    telemetry_source_path = root / "src/terraclass/telemetry.py"
    telemetry_test_path = root / "tests/test_telemetry.py"
    monitoring_config_path = root / "configs/monitoring/observability_v1.json"
    monitoring_5xx_policy_path = root / "deploy/monitoring/cloud-run-5xx-ratio.json"
    monitoring_latency_policy_path = root / "deploy/monitoring/cloud-run-p95-latency.json"
    pyproject_path = root / "pyproject.toml"
    web_app_path = root / "web/app/TerraClassApp.tsx"
    web_css_path = root / "web/app/globals.css"
    web_package_path = root / "web/package.json"
    web_test_path = root / "web/tests/rendered-html.test.mjs"
    web_preview_source_path = root / "web/lib/image-preview.ts"
    web_preview_test_path = root / "web/tests/image-preview.test.mjs"
    web_hosting_path = root / "web/.openai/hosting.json"
    web_vercel_path = root / "web/vercel.json"
    dockerfile_path = root / "Dockerfile"
    dockerignore_path = root / ".dockerignore"
    cloud_run_template_path = root / "deploy/cloud-run-service.template.yaml"
    ci_workflow_path = root / ".github/workflows/ci.yml"
    container_workflow_path = root / ".github/workflows/container-release.yml"

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

    model_release: ModelRelease | None = None
    if not model_release_path.is_file():
        report.errors.append("Versioned model-release contract is missing")
    else:
        try:
            model_release = load_model_release(model_release_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            report.errors.append(f"Model-release contract is invalid: {error}")
        if model_release is not None and serving_config is not None:
            if (
                model_release.model_id,
                model_release.model_version,
                model_release.sha256,
            ) != (
                serving_config.model_id,
                serving_config.model_version,
                serving_config.serving_artifact.sha256,
            ):
                report.errors.append("Model-release identity differs from serving config")
            if model_release.asset_name != Path(serving_config.serving_artifact.path).name:
                report.errors.append("Model-release asset name differs from serving config")
            if serving_artifact_hash is not None:
                serving_artifact_path = root / serving_config.serving_artifact.path
                if serving_artifact_path.stat().st_size != model_release.size_bytes:
                    report.errors.append(
                        "Local serving artifact size differs from release contract"
                    )

    if not model_release_verification_path.is_file():
        report.errors.append("Public model-release verification report is missing")
        model_release_verification: dict[str, Any] = {}
    else:
        try:
            model_release_verification = json.loads(
                model_release_verification_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            report.errors.append(f"Model-release verification report is invalid: {error}")
            model_release_verification = {}
        if model_release_verification.get("schema_version") != 1:
            report.errors.append("Model-release verification schema version differs from 1")
        if model_release_verification.get("verified_on") != "2026-07-16":
            report.errors.append("Model-release verification date differs from 16 July 2026")
        repository_evidence = model_release_verification.get("repository", {})
        if repository_evidence != {
            "url": "https://github.com/joydas65/land-use-classification",
            "visibility": "public",
        }:
            report.errors.append("Public repository evidence differs from the release target")
        release_evidence = model_release_verification.get("release", {})
        if (
            release_evidence.get("url")
            != ("https://github.com/joydas65/land-use-classification/releases/tag/model-v1.0.0")
            or release_evidence.get("tag") != "model-v1.0.0"
        ):
            report.errors.append("Published release evidence differs from model-v1.0.0")
        if re.fullmatch(r"[0-9a-f]{40}", str(release_evidence.get("target_commit"))) is None:
            report.errors.append("Model-release target commit is not a full Git SHA")
        asset_evidence = model_release_verification.get("asset", {})
        if model_release is not None and asset_evidence != {
            "name": model_release.asset_name,
            "url": model_release.url,
            "size_bytes": model_release.size_bytes,
            "sha256": model_release.sha256,
        }:
            report.errors.append("Published model asset differs from the release contract")
        if model_release_verification.get("verification") != {
            "method": "fresh unauthenticated HTTPS download",
            "public_access": True,
            "size_matches_contract": True,
            "sha256_matches_contract": True,
        }:
            report.errors.append("Public model-download verification is incomplete")

    if not container_release_verification_path.is_file():
        report.errors.append("Public container-release verification report is missing")
        container_release_verification: dict[str, Any] = {}
    else:
        try:
            container_release_verification = json.loads(
                container_release_verification_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            report.errors.append(f"Container-release verification report is invalid: {error}")
            container_release_verification = {}
        if container_release_verification.get("schema_version") != 1:
            report.errors.append("Container-release verification schema version differs from 1")
        if container_release_verification.get("verified_on") != "2026-07-16":
            report.errors.append("Container-release verification date differs from 16 July 2026")
        source_evidence = container_release_verification.get("source", {})
        if (
            source_evidence.get("repository")
            != ("https://github.com/joydas65/land-use-classification")
            or source_evidence.get("tag") != "api-v1.0.0"
        ):
            report.errors.append("Container source evidence differs from the release tag")
        if re.fullmatch(r"[0-9a-f]{40}", str(source_evidence.get("commit"))) is None:
            report.errors.append("Container source commit is not a full Git SHA")
        workflow_evidence = container_release_verification.get("workflow", {})
        if workflow_evidence.get("run_id") != 29503393345 or (
            workflow_evidence.get("conclusion") != "success"
        ):
            report.errors.append("Container-release workflow evidence is not successful")
        registry_evidence = container_release_verification.get("registry", {})
        if (
            registry_evidence.get("repository") != "ghcr.io/joydas65/terraclass-api"
            or registry_evidence.get("visibility") != "public"
            or registry_evidence.get("public_pull_verified") is not True
        ):
            report.errors.append("Public GHCR pull evidence is incomplete")
        tag_digests = registry_evidence.get("tags", {})
        index_evidence = container_release_verification.get("oci_index", {})
        index_digest = index_evidence.get("digest")
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(index_digest)) is None
            or tag_digests.get("api-v1.0.0") != index_digest
            or tag_digests.get("sha-3b5b074") != index_digest
        ):
            report.errors.append("Container tags do not share one immutable OCI digest")
        image_evidence = container_release_verification.get("image", {})
        image_digest = image_evidence.get("digest")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(image_digest)) is None or image_evidence.get(
            "platform"
        ) != {"architecture": "amd64", "os": "linux"}:
            report.errors.append("Container platform-image evidence is invalid")
        attestation_evidence = container_release_verification.get("attestation_manifest", {})
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(attestation_evidence.get("digest"))) is None
            or attestation_evidence.get("subject_digest") != image_digest
        ):
            report.errors.append("Container attestation manifest does not bind the image")
        attestation_layers = attestation_evidence.get("layers", [])
        if {
            layer.get("predicate_type") for layer in attestation_layers if isinstance(layer, dict)
        } != {"https://spdx.dev/Document", "https://slsa.dev/provenance/v1"}:
            report.errors.append("Container release must contain SPDX and SLSA attestations")
        for layer in attestation_layers:
            if not isinstance(layer, dict) or (
                re.fullmatch(r"sha256:[0-9a-f]{64}", str(layer.get("digest"))) is None
                or not isinstance(layer.get("size_bytes"), int)
                or layer.get("size_bytes", 0) <= 0
            ):
                report.errors.append("Container attestation layer evidence is invalid")

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

    load_levels: list[dict[str, Any]] = []
    if not api_load_report_path.is_file():
        report.errors.append("Versioned HTTP load report is missing")
        api_load_report: dict[str, Any] = {}
    else:
        api_load_report = json.loads(api_load_report_path.read_text(encoding="utf-8"))
        if api_load_report.get("schema_version") != 1:
            report.errors.append("HTTP load report schema version differs from 1")
        load_model = api_load_report.get("model", {})
        if serving_config is not None and load_model != {
            "model_id": serving_config.model_id,
            "model_version": serving_config.model_version,
            "serving_artifact_sha256": serving_config.serving_artifact.sha256,
        }:
            report.errors.append("HTTP load report model identity differs from serving config")
        load_protocol = api_load_report.get("protocol", {})
        if load_protocol.get("concurrency_levels") != [1, 2, 4]:
            report.errors.append("HTTP load report must cover concurrency 1, 2, and 4")
        if load_protocol.get("warmup_requests") != 5:
            report.errors.append("HTTP load report must contain five warm-up requests")
        if load_protocol.get("requests_per_level") != 20:
            report.errors.append("HTTP load report must contain 20 requests per level")
        load_levels = api_load_report.get("levels", [])
        if len(load_levels) != 3:
            report.errors.append("HTTP load report must contain three concurrency levels")
        elif any(level.get("failures") != 0 for level in load_levels):
            report.errors.append("HTTP load report contains a failed request")
        for level in load_levels:
            if level.get("requests") != 20:
                report.errors.append("HTTP load level does not contain 20 measured requests")
            latency = level.get("total_latency_ms", {})
            if not 0 < latency.get("p50", 0) <= latency.get("p95", 0):
                report.errors.append("HTTP load level p50/p95 latency is inconsistent")
            if level.get("throughput_requests_per_second", 0) <= 0:
                report.errors.append("HTTP load level throughput must be positive")

    cloud_load_levels: list[dict[str, Any]] = []
    if not cloud_run_load_report_path.is_file():
        report.errors.append("Versioned Cloud Run load report is missing")
        cloud_run_load_report: dict[str, Any] = {}
    else:
        cloud_run_load_report = json.loads(cloud_run_load_report_path.read_text(encoding="utf-8"))
        if cloud_run_load_report.get("schema_version") != 1:
            report.errors.append("Cloud Run load report schema version differs from 1")
        if (
            cloud_run_load_report.get("target")
            != "https://terraclass-api-280836764570.asia-south1.run.app"
        ):
            report.errors.append("Cloud Run load report target differs from production")
        cloud_load_model = cloud_run_load_report.get("model", {})
        if serving_config is not None and cloud_load_model != {
            "model_id": serving_config.model_id,
            "model_version": serving_config.model_version,
            "serving_artifact_sha256": serving_config.serving_artifact.sha256,
        }:
            report.errors.append("Cloud Run load model identity differs from serving config")
        cloud_load_protocol = cloud_run_load_report.get("protocol", {})
        if cloud_load_protocol.get("concurrency_levels") != [1, 2, 4]:
            report.errors.append("Cloud Run load report must cover concurrency 1, 2, and 4")
        if cloud_load_protocol.get("warmup_requests") != 5:
            report.errors.append("Cloud Run load report must contain five warm-up requests")
        if cloud_load_protocol.get("requests_per_level") != 20:
            report.errors.append("Cloud Run load report must contain 20 requests per level")
        cloud_load_levels = cloud_run_load_report.get("levels", [])
        if len(cloud_load_levels) != 3:
            report.errors.append("Cloud Run load report must contain three concurrency levels")
        elif any(level.get("failures") != 0 for level in cloud_load_levels):
            report.errors.append("Cloud Run load report contains a failed request")
        for level in cloud_load_levels:
            latency = level.get("total_latency_ms", {})
            if level.get("requests") != 20:
                report.errors.append("Cloud Run load level does not contain 20 measured requests")
            if not 0 < latency.get("p50", 0) <= latency.get("p95", 0):
                report.errors.append("Cloud Run load level p50/p95 latency is inconsistent")
            if level.get("throughput_requests_per_second", 0) <= 0:
                report.errors.append("Cloud Run load level throughput must be positive")

    if not cloud_run_deployment_verification_path.is_file():
        report.errors.append("Cloud Run deployment verification report is missing")
        cloud_run_deployment: dict[str, Any] = {}
    else:
        cloud_run_deployment = json.loads(
            cloud_run_deployment_verification_path.read_text(encoding="utf-8")
        )
        if cloud_run_deployment.get("schema_version") != 1:
            report.errors.append("Cloud Run deployment schema version differs from 1")
        if cloud_run_deployment.get("verified_on") != "2026-07-16":
            report.errors.append("Cloud Run deployment date differs from 16 July 2026")
        cloud_run = cloud_run_deployment.get("cloud_run", {})
        if {
            "project_id": cloud_run.get("project_id"),
            "project_number": cloud_run.get("project_number"),
            "region": cloud_run.get("region"),
            "service": cloud_run.get("service"),
            "revision": cloud_run.get("revision"),
            "service_url": cloud_run.get("service_url"),
            "ready": cloud_run.get("ready"),
            "public_access": cloud_run.get("public_access"),
            "traffic_percent": cloud_run.get("traffic_percent"),
        } != {
            "project_id": "land-use-classification-502614",
            "project_number": "280836764570",
            "region": "asia-south1",
            "service": "terraclass-api",
            "revision": "terraclass-api-v1-0-1",
            "service_url": "https://terraclass-api-280836764570.asia-south1.run.app",
            "ready": True,
            "public_access": True,
            "traffic_percent": 100,
        }:
            report.errors.append("Cloud Run service identity/readiness evidence is inconsistent")
        if cloud_run.get("deployed_oci_index_digest") != container_release_verification.get(
            "oci_index", {}
        ).get("digest") or cloud_run.get(
            "resolved_linux_amd64_digest"
        ) != container_release_verification.get("image", {}).get("digest"):
            report.errors.append("Cloud Run deployment does not bind the released container")
        if cloud_run.get("runtime_identity") != {
            "service_account": (
                "terraclass-runtime@land-use-classification-502614.iam.gserviceaccount.com"
            ),
            "project_roles": [],
            "replaced_default_identity_role": "roles/editor",
        }:
            report.errors.append("Cloud Run runtime identity is not least-privilege evidence")
        if (
            cloud_run.get("public_invoker_role") != "roles/run.invoker"
            or cloud_run.get("public_invoker_member") != "allUsers"
        ):
            report.errors.append("Cloud Run public invoker evidence is inconsistent")
        if cloud_run.get("resources") != {
            "cpu": "2",
            "memory": "2Gi",
            "container_concurrency": 4,
            "min_instances": 0,
            "max_instances": 3,
            "timeout_seconds": 30,
            "cpu_throttling": True,
            "startup_cpu_boost": True,
        }:
            report.errors.append("Cloud Run resources differ from the production contract")
        if cloud_run.get("capacity") != {
            "max_concurrent_inferences": 1,
            "queue_timeout_seconds": 5,
        }:
            report.errors.append("Cloud Run inference capacity differs from the API contract")
        if (
            cloud_run.get("cors_allowed_origin")
            != "https://terraclass-land-use-classification.vercel.app"
        ):
            report.errors.append("Cloud Run CORS origin differs from the production frontend")
        initial_rollout = cloud_run.get("initial_rollout", {})
        for duration_name in (
            "revision_ready_seconds",
            "container_healthy_seconds",
        ):
            if initial_rollout.get(duration_name, 0) <= 0:
                report.errors.append(f"Cloud Run rollout duration is invalid: {duration_name}")
        if initial_rollout.get("image_import_cached") is not True:
            report.errors.append("Hardened Cloud Run revision did not record cached image import")
        if initial_rollout.get("scale_to_zero_cold_request_measured") is not False:
            report.errors.append("Cloud Run cold-request claim boundary is inconsistent")
        cloud_api = cloud_run_deployment.get("api_verification", {})
        if serving_config is not None and {
            "readiness_status_code": cloud_api.get("readiness_status_code"),
            "model_metadata_status_code": cloud_api.get("model_metadata_status_code"),
            "model_id": cloud_api.get("model_id"),
            "model_version": cloud_api.get("model_version"),
            "serving_artifact_sha256": cloud_api.get("serving_artifact_sha256"),
        } != {
            "readiness_status_code": 200,
            "model_metadata_status_code": 200,
            "model_id": serving_config.model_id,
            "model_version": serving_config.model_version,
            "serving_artifact_sha256": serving_config.serving_artifact.sha256,
        }:
            report.errors.append("Cloud Run API identity/health evidence is inconsistent")
        prediction = cloud_api.get("prediction", {})
        if (
            prediction.get("status_code") != 200
            or prediction.get("expected_class") != "agricultural"
            or prediction.get("predicted_class") != "agricultural"
            or not 0 < prediction.get("confidence", 0) <= 1
        ):
            report.errors.append("Cloud Run production prediction evidence is invalid")
        cors = cloud_api.get("cors_preflight", {})
        if (
            cors.get("status_code") != 200
            or cors.get("allowed_origin") != "https://terraclass-land-use-classification.vercel.app"
            or set(cors.get("allowed_methods", [])) != {"GET", "POST", "OPTIONS"}
        ):
            report.errors.append("Cloud Run production CORS evidence is invalid")
        cloud_measured_requests = sum(int(level.get("requests", 0)) for level in cloud_load_levels)
        cloud_failures = sum(int(level.get("failures", 0)) for level in cloud_load_levels)
        cloud_peak_throughput = max(
            (level.get("throughput_requests_per_second", 0) for level in cloud_load_levels),
            default=0,
        )
        cloud_concurrency_4_p95 = next(
            (
                level.get("total_latency_ms", {}).get("p95")
                for level in cloud_load_levels
                if level.get("concurrency") == 4
            ),
            None,
        )
        if (
            cloud_api.get("load_report") != "reports/cloud_run_load_test_2026-07-16.json"
            or cloud_api.get("warmup_requests") != 5
            or cloud_api.get("measured_requests") != cloud_measured_requests
            or cloud_api.get("failures") != cloud_failures
            or cloud_api.get("peak_throughput_requests_per_second") != cloud_peak_throughput
            or cloud_api.get("concurrency_4_p95_total_latency_ms") != cloud_concurrency_4_p95
        ):
            report.errors.append("Cloud Run load summary differs from the versioned load report")
        vercel = cloud_run_deployment.get("vercel", {})
        if {
            "project": vercel.get("project"),
            "production_alias": vercel.get("production_alias"),
            "target": vercel.get("target"),
            "status": vercel.get("status"),
            "api_environment_variable": vercel.get("api_environment_variable"),
            "browser_model_status": vercel.get("browser_model_status"),
            "browser_console_errors": vercel.get("browser_console_errors"),
            "browser_console_warnings": vercel.get("browser_console_warnings"),
        } != {
            "project": "terraclass-land-use-classification",
            "production_alias": "https://terraclass-land-use-classification.vercel.app",
            "target": "production",
            "status": "ready",
            "api_environment_variable": "NEXT_PUBLIC_TERRACLASS_API_URL",
            "browser_model_status": "Model ready",
            "browser_console_errors": 0,
            "browser_console_warnings": 0,
        }:
            report.errors.append("Vercel integration evidence is inconsistent")
        if cloud_run_deployment.get("claim_boundary") != {
            "cloud_run_api_deployed": True,
            "integrated_public_classifier_deployed": True,
            "production_load_probe_completed": True,
            "production_slo_established": False,
            "scale_to_zero_cold_request_measured": False,
        }:
            report.errors.append("Integrated deployment claim boundary is inconsistent")

    if not cloud_run_scale_to_zero_path.is_file():
        report.errors.append("Cloud Run scale-to-zero evidence is missing")
        cloud_run_scale_to_zero: dict[str, Any] = {}
    else:
        cloud_run_scale_to_zero = json.loads(
            cloud_run_scale_to_zero_path.read_text(encoding="utf-8")
        )
        cold_service = cloud_run_scale_to_zero.get("service", {})
        cold_precondition = cloud_run_scale_to_zero.get("scale_to_zero_precondition", {})
        cold_probe = cloud_run_scale_to_zero.get("client_probe", {})
        cold_response = cold_probe.get("response", {})
        cold_corroboration = cloud_run_scale_to_zero.get("cloud_run_corroboration", {})
        cold_claim = cloud_run_scale_to_zero.get("claim_boundary", {})
        if cloud_run_scale_to_zero.get("schema_version") != 1:
            report.errors.append("Cloud Run scale-to-zero schema version differs from 1")
        if cloud_run_scale_to_zero.get("measured_on") != "2026-07-17":
            report.errors.append("Cloud Run scale-to-zero date differs from 17 July 2026")
        if {
            "project_id": cold_service.get("project_id"),
            "region": cold_service.get("region"),
            "service_name": cold_service.get("service_name"),
            "revision": cold_service.get("revision"),
            "minimum_instances": cold_service.get("minimum_instances"),
        } != {
            "project_id": "land-use-classification-502614",
            "region": "asia-south1",
            "service_name": "terraclass-api",
            "revision": "terraclass-api-v1-0-1",
            "minimum_instances": 0,
        }:
            report.errors.append("Cloud Run scale-to-zero service identity is inconsistent")
        if (
            cold_precondition.get("request_gap_seconds", 0)
            <= cold_precondition.get("documented_possible_idle_retention_seconds", 900)
            or cold_precondition.get("active_instances") != 0
            or cold_precondition.get("idle_instances") != 0
        ):
            report.errors.append("Cloud Run scale-to-zero precondition is not proven")
        if (
            cold_probe.get("http_status") != 200
            or cold_response.get("expected_class") != "agricultural"
            or cold_response.get("predicted_class") != "agricultural"
            or cold_response.get("inference_latency_ms", 0) <= 0
            or cold_probe.get("curl_timings_ms", {}).get("total", 0) <= 0
        ):
            report.errors.append("Cloud Run cold client prediction evidence is invalid")
        if (
            cold_corroboration.get("request_log", {}).get("http_status") != 200
            or cold_corroboration.get("instance_start_log", {}).get("matches_request_instance")
            is not True
            or "AUTOSCALING"
            not in cold_corroboration.get("instance_start_log", {}).get("reason", "")
        ):
            report.errors.append("Cloud Run cold autoscaling corroboration is invalid")
        if cold_claim != {
            "scale_from_zero_client_request_measured": True,
            "correct_prediction_observed": True,
            "availability_slo_established": False,
            "latency_slo_established": False,
            "semantic_drift_evaluated": False,
            "note": (
                "This is one point-in-time cold request, not a percentile, uptime guarantee, "
                "or drift result."
            ),
        }:
            report.errors.append("Cloud Run cold-request claim boundary is inconsistent")

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
        "InferenceCapacityError",
        "asyncio.Semaphore",
        "TERRACLASS_MAX_CONCURRENT_INFERENCES",
        "prediction_observation",
        "emit_structured_event",
        "access_log=False",
    ):
        if token not in api_source:
            report.errors.append(f"API safety/operability token is missing: {token}")

    api_test_count = 0
    if not api_test_path.is_file():
        report.errors.append("API contract tests are missing")
    else:
        api_test_source = api_test_path.read_text(encoding="utf-8")
        api_test_count = len(re.findall(r"^def test_", api_test_source, re.MULTILINE))
        if api_test_count != 7:
            report.errors.append("API contract suite must contain seven focused tests")

    telemetry_test_count = 0
    if not telemetry_source_path.is_file():
        report.errors.append("Prediction telemetry source is missing")
    if not telemetry_test_path.is_file():
        report.errors.append("Prediction telemetry tests are missing")
    else:
        telemetry_test_source = telemetry_test_path.read_text(encoding="utf-8")
        telemetry_test_count = len(re.findall(r"^def test_", telemetry_test_source, re.MULTILINE))
        if telemetry_test_count != 4:
            report.errors.append("Prediction telemetry suite must contain four focused tests")

    if not monitoring_config_path.is_file():
        report.errors.append("Machine-readable observability contract is missing")
        monitoring_config: dict[str, Any] = {}
    else:
        monitoring_config = json.loads(monitoring_config_path.read_text(encoding="utf-8"))
        monitoring_service = monitoring_config.get("service", {})
        telemetry_contract = monitoring_config.get("telemetry", {})
        candidate_objectives = monitoring_config.get("candidate_objectives", {})
        drift_readiness = monitoring_config.get("drift_readiness", {})
        if monitoring_config.get("schema_version") != 1:
            report.errors.append("Observability schema version differs from 1")
        if monitoring_service != {
            "project_id": "land-use-classification-502614",
            "region": "asia-south1",
            "service_name": "terraclass-api",
            "service_version": "1.1.0",
            "model_id": "terraclass-resnet18-group-aware",
            "model_version": "1.0.0",
        }:
            report.errors.append("Observability service/model identity is inconsistent")
        if telemetry_contract.get("allowlisted_fields") != list(PREDICTION_OBSERVATION_FIELDS):
            report.errors.append("Prediction telemetry allowlist differs from code")
        if set(telemetry_contract.get("prohibited_fields", [])) != set(
            PROHIBITED_PREDICTION_FIELDS
        ):
            report.errors.append("Prediction telemetry prohibited fields differ from code")
        if telemetry_contract.get("image_content_retained") is not False:
            report.errors.append("Prediction telemetry must not retain image content")
        if candidate_objectives.get("status") != "defined_not_yet_established":
            report.errors.append("Candidate objectives are incorrectly marked as established")
        if drift_readiness.get("status") != "telemetry_ready_not_drift_validated":
            report.errors.append("Drift readiness claim boundary is inconsistent")

    expected_monitoring_policies = {
        "deploy/monitoring/cloud-run-5xx-ratio.json": (
            monitoring_5xx_policy_path,
            "run.googleapis.com/request_count",
            "availability",
        ),
        "deploy/monitoring/cloud-run-p95-latency.json": (
            monitoring_latency_policy_path,
            "run.googleapis.com/request_latencies",
            "steady-state-latency",
        ),
    }
    if set(monitoring_config.get("alert_policy_templates", [])) != set(
        expected_monitoring_policies
    ):
        report.errors.append("Observability alert policy list is inconsistent")
    for relative_path, (policy_path, metric, objective) in expected_monitoring_policies.items():
        if not policy_path.is_file():
            report.errors.append(f"Monitoring policy is missing: {relative_path}")
            continue
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("enabled") is not True:
            report.errors.append(f"Monitoring policy is disabled: {relative_path}")
        if policy.get("userLabels", {}).get("objective") != objective:
            report.errors.append(f"Monitoring policy objective differs: {relative_path}")
        if metric not in json.dumps(policy) or "terraclass-api" not in json.dumps(policy):
            report.errors.append(f"Monitoring policy target differs: {relative_path}")

    if not cloud_monitoring_deployment_path.is_file():
        report.errors.append("Cloud Monitoring deployment evidence is missing")
        cloud_monitoring_deployment: dict[str, Any] = {}
    else:
        cloud_monitoring_deployment = json.loads(
            cloud_monitoring_deployment_path.read_text(encoding="utf-8")
        )
        deployed_policies = cloud_monitoring_deployment.get("policies", [])
        monitoring_claim = cloud_monitoring_deployment.get("claim_boundary", {})
        if (
            cloud_monitoring_deployment.get("schema_version") != 1
            or cloud_monitoring_deployment.get("verified_on") != "2026-07-17"
            or cloud_monitoring_deployment.get("project_id") != "land-use-classification-502614"
            or cloud_monitoring_deployment.get("readback_verified") is not True
        ):
            report.errors.append("Cloud Monitoring deployment identity is inconsistent")
        if len(deployed_policies) != 2 or {
            policy.get("template") for policy in deployed_policies
        } != set(expected_monitoring_policies):
            report.errors.append("Cloud Monitoring deployed policy set is inconsistent")
        for deployed_policy in deployed_policies:
            if (
                deployed_policy.get("enabled") is not True
                or not str(deployed_policy.get("name", "")).startswith(
                    "projects/land-use-classification-502614/alertPolicies/"
                )
                or deployed_policy.get("notification_channels") != []
            ):
                report.errors.append("Cloud Monitoring policy readback is inconsistent")
        if monitoring_claim != {
            "alert_policies_deployed": True,
            "incident_creation_enabled": True,
            "notifications_routed": False,
            "candidate_objectives_established_as_slo": False,
            "note": (
                "A notification channel must be chosen and verified separately; deployed policies "
                "do not create 30-day historical evidence."
            ),
        }:
            report.errors.append("Cloud Monitoring deployment claim boundary is inconsistent")

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    web_dependencies = pyproject.get("project", {}).get("optional-dependencies", {}).get("web", [])
    if not any(str(dependency).startswith("fastapi") for dependency in web_dependencies):
        report.errors.append("FastAPI is missing from the web optional dependency set")
    if pyproject.get("project", {}).get("scripts", {}).get("terraclass-api") != (
        "terraclass.api:main"
    ):
        report.errors.append("terraclass-api command does not target terraclass.api:main")

    deployment_contracts = (
        (
            dockerfile_path,
            (
                "AS runtime-base",
                "https://download.pytorch.org/whl/cpu",
                "USER terraclass",
                "fetch_serving_artifact.py",
                "HEALTHCHECK",
            ),
        ),
        (dockerignore_path, ("artifacts", "data/raw", "*.pt")),
        (
            cloud_run_template_path,
            (
                "containerConcurrency: 4",
                "memory: 2Gi",
                "serviceAccountName: terraclass-runtime@PROJECT_ID.iam.gserviceaccount.com",
                "TERRACLASS_MAX_CONCURRENT_INFERENCES",
                "/api/v1/health/ready",
            ),
        ),
        (
            ci_workflow_path,
            (
                "audit_consistency.py",
                "python -m pip_audit",
                "setuptools>=83",
                "pytest -q",
                "npm test",
                "target: runtime-base",
            ),
        ),
        (
            container_workflow_path,
            ("registry: ghcr.io", "target: production", "provenance: mode=max", "sbom: true"),
        ),
    )
    for path, tokens in deployment_contracts:
        if not path.is_file():
            report.errors.append(f"Deployment contract is missing: {path.relative_to(root)}")
            continue
        contract = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in contract:
                report.errors.append(
                    f"Deployment contract token is missing from {path.relative_to(root)}: {token}"
                )

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
        "createImagePreviewUrl",
        "Preparing preview",
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
        if web_package.get("engines", {}).get("node") != "24.x":
            report.errors.append("Browser Node.js runtime is not pinned to the audited major")
        if web_package.get("scripts", {}).get("test") != (
            "vinext build && node --test tests/rendered-html.test.mjs tests/image-preview.test.mjs"
        ):
            report.errors.append("Browser test command does not build before render testing")
        if web_package.get("dependencies", {}).get("tiff") != "7.1.3":
            report.errors.append("Browser TIFF decoder is not pinned to the audited version")
        if web_package.get("scripts", {}).get("build:vercel") != "next build":
            report.errors.append("Vercel build command does not use the native Next.js build")
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
    web_preview_test_count = 0
    if not web_preview_source_path.is_file():
        report.errors.append("Browser TIFF preview implementation is missing")
    else:
        web_preview_source = web_preview_source_path.read_text(encoding="utf-8")
        for token in ("decode(new Uint8Array", "pages: [0]", '"image/png"', "25_000_000"):
            if token not in web_preview_source:
                report.errors.append(f"Browser TIFF preview contract token is missing: {token}")
    if not web_preview_test_path.is_file():
        report.errors.append("Browser TIFF preview tests are missing")
    else:
        web_preview_test_source = web_preview_test_path.read_text(encoding="utf-8")
        web_preview_test_count = len(re.findall(r'^test\("', web_preview_test_source, re.MULTILINE))
        if web_preview_test_count != 3:
            report.errors.append("Browser TIFF preview suite must contain three focused tests")
    if not web_css_path.is_file():
        report.errors.append("Browser global stylesheet is missing")
        web_css_source = ""
    else:
        web_css_source = web_css_path.read_text(encoding="utf-8")
        for token in ('@import "tailwindcss"', "@theme"):
            if token not in web_css_source:
                report.errors.append(f"Tailwind CSS contract token is missing: {token}")
    for token in ("bg-paper", "text-teal", "max-[620px]:grid-cols-2"):
        if token not in web_app_source:
            report.errors.append(f"Tailwind utility token is missing from browser source: {token}")
    if not web_hosting_path.is_file():
        report.errors.append("Sites hosting configuration is missing")
    else:
        hosting_config = json.loads(web_hosting_path.read_text(encoding="utf-8"))
        unexpected_hosting_keys = set(hosting_config) - {"project_id", "d1", "r2"}
        if unexpected_hosting_keys:
            report.errors.append("Sites hosting configuration contains unexpected keys")
    if not web_vercel_path.is_file():
        report.errors.append("Vercel project configuration is missing")
    else:
        vercel_config = json.loads(web_vercel_path.read_text(encoding="utf-8"))
        expected_vercel_config = {
            "$schema": "https://openapi.vercel.sh/vercel.json",
            "framework": "nextjs",
            "buildCommand": "npm run build:vercel",
            "installCommand": "npm ci",
        }
        if vercel_config != expected_vercel_config:
            report.errors.append("Vercel project configuration differs from the audited contract")

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
    if not production_document_path.is_file():
        report.errors.append("docs/PRODUCTION_INFERENCE.md is missing")
        production_document = ""
    else:
        production_document = production_document_path.read_text(encoding="utf-8")
    if not observability_document_path.is_file():
        report.errors.append("docs/OBSERVABILITY_AND_DRIFT.md is missing")
        observability_document = ""
    else:
        observability_document = observability_document_path.read_text(encoding="utf-8")
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
            production_document,
            observability_document,
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
    api_document_normalized = " ".join(api_document.split())
    for token in (
        "FastAPI",
        "request ID",
        "NEXT_PUBLIC_TERRACLASS_API_URL",
        "https://terraclass-land-use-classification.vercel.app",
        "Tailwind CSS",
        "deployed integrated system",
        "https://terraclass-api-280836764570.asia-south1.run.app",
        "Model ready",
    ):
        if token not in api_document_normalized:
            report.errors.append(f"Application documentation token is missing: {token}")
    production_document_normalized = " ".join(production_document.split())
    for token in (
        "16 July 2026",
        "60 measured requests",
        "52.9 requests/second",
        "84.1 ms",
        "model-v1.0.0",
        "44,795,275 bytes",
        "fresh unauthenticated HTTPS download",
        "29457675941",
        "29503393345",
        "sha256:484766fe9334a2807813edbdee0bfe637d71bac2af60c78a9642a807201ccd73",
        "SPDX",
        "SLSA provenance v1",
        "land-use-classification-502614",
        "terraclass-api-v1-0-1",
        "https://terraclass-api-280836764570.asia-south1.run.app",
        "13.3 requests/second",
        "365.2 ms",
        "scale-to-zero",
        "terraclass-runtime@land-use-classification-502614.iam.gserviceaccount.com",
        "dpl_A7JEXaCo8BeHK5v7drCUFMeekWop",
        "17 July scale-to-zero and monitoring extension",
        "1,119.990-second",
        "11,013.115 ms",
        "321.804 ms",
        "cloud_run_scale_to_zero_2026-07-17.json",
        "cloud_monitoring_deployment_2026-07-17.json",
    ):
        if token not in production_document_normalized:
            report.errors.append(f"Production documentation token is missing: {token}")
    observability_document_normalized = " ".join(observability_document.split())
    for token in (
        "prediction_observation",
        "does **not** log the uploaded filename",
        "99% over a rolling 30-day window",
        "p95 at or below 1,000 ms",
        "defined_not_yet_established",
        "365.2 ms",
        "11,013.115 ms",
        "5310080937064810576",
        "5310080937064809962",
        "drift-*ready*, not a validated drift detector",
    ):
        if token not in "\n".join(
            (observability_document_normalized, json.dumps(monitoring_config))
        ):
            report.errors.append(f"Observability documentation token is missing: {token}")

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
        "production_readiness": {
            "model_release_url": model_release.url if model_release else None,
            "model_release_size_bytes": model_release.size_bytes if model_release else None,
            "model_release_public_download_verified": model_release_verification.get(
                "verification", {}
            ).get("public_access"),
            "model_release_target_commit": model_release_verification.get("release", {}).get(
                "target_commit"
            ),
            "container_release_workflow_run": container_release_verification.get(
                "workflow", {}
            ).get("run_id"),
            "container_image": container_release_verification.get("registry", {}).get("repository"),
            "container_index_digest": container_release_verification.get("oci_index", {}).get(
                "digest"
            ),
            "container_platform_digest": container_release_verification.get("image", {}).get(
                "digest"
            ),
            "container_public_pull_verified": container_release_verification.get(
                "registry", {}
            ).get("public_pull_verified"),
            "container_sbom_attested": "https://spdx.dev/Document"
            in {
                layer.get("predicate_type")
                for layer in container_release_verification.get("attestation_manifest", {}).get(
                    "layers", []
                )
                if isinstance(layer, dict)
            },
            "container_provenance_attested": "https://slsa.dev/provenance/v1"
            in {
                layer.get("predicate_type")
                for layer in container_release_verification.get("attestation_manifest", {}).get(
                    "layers", []
                )
                if isinstance(layer, dict)
            },
            "container_contract": dockerfile_path.is_file(),
            "ci_workflows": ci_workflow_path.is_file() and container_workflow_path.is_file(),
            "local_http_warmup_requests": api_load_report.get("protocol", {}).get(
                "warmup_requests"
            ),
            "local_http_measured_requests": sum(
                int(level.get("requests", 0)) for level in load_levels
            ),
            "local_http_concurrency_levels": [level.get("concurrency") for level in load_levels],
            "local_http_peak_throughput_rps": max(
                (level.get("throughput_requests_per_second", 0) for level in load_levels),
                default=0,
            ),
            "local_http_concurrency_4_p95_ms": next(
                (
                    level.get("total_latency_ms", {}).get("p95")
                    for level in load_levels
                    if level.get("concurrency") == 4
                ),
                None,
            ),
            "production_api_deployed": cloud_run_deployment.get("claim_boundary", {}).get(
                "cloud_run_api_deployed"
            ),
            "cloud_run_service_url": cloud_run_deployment.get("cloud_run", {}).get("service_url"),
            "cloud_run_revision": cloud_run_deployment.get("cloud_run", {}).get("revision"),
            "cloud_run_region": cloud_run_deployment.get("cloud_run", {}).get("region"),
            "cloud_run_rollout_container_healthy_seconds": cloud_run_deployment.get("cloud_run", {})
            .get("initial_rollout", {})
            .get("container_healthy_seconds"),
            "cloud_run_scale_to_zero_cold_request_measured": cloud_run_scale_to_zero.get(
                "claim_boundary", {}
            ).get("scale_from_zero_client_request_measured"),
            "cloud_run_scale_to_zero_client_total_ms": cloud_run_scale_to_zero.get(
                "client_probe", {}
            )
            .get("curl_timings_ms", {})
            .get("total"),
            "cloud_run_http_warmup_requests": cloud_run_load_report.get("protocol", {}).get(
                "warmup_requests"
            ),
            "cloud_run_http_measured_requests": sum(
                int(level.get("requests", 0)) for level in cloud_load_levels
            ),
            "cloud_run_http_failures": sum(
                int(level.get("failures", 0)) for level in cloud_load_levels
            ),
            "cloud_run_http_peak_throughput_rps": max(
                (level.get("throughput_requests_per_second", 0) for level in cloud_load_levels),
                default=0,
            ),
            "cloud_run_http_concurrency_4_p95_ms": next(
                (
                    level.get("total_latency_ms", {}).get("p95")
                    for level in cloud_load_levels
                    if level.get("concurrency") == 4
                ),
                None,
            ),
            "production_slo_established": cloud_run_deployment.get("claim_boundary", {}).get(
                "production_slo_established"
            ),
            "prediction_telemetry_fields": monitoring_config.get("telemetry", {}).get(
                "allowlisted_fields"
            ),
            "candidate_availability_target": monitoring_config.get("candidate_objectives", {})
            .get("availability", {})
            .get("target_ratio"),
            "candidate_p95_latency_ms": monitoring_config.get("candidate_objectives", {})
            .get("steady_state_request_latency", {})
            .get("threshold_ms"),
            "drift_readiness_status": monitoring_config.get("drift_readiness", {}).get("status"),
            "cloud_monitoring_policies_deployed": cloud_monitoring_deployment.get(
                "claim_boundary", {}
            ).get("alert_policies_deployed"),
            "cloud_monitoring_notifications_routed": cloud_monitoring_deployment.get(
                "claim_boundary", {}
            ).get("notifications_routed"),
        },
        "application_layer": {
            "api_routes": list(expected_api_routes),
            "api_contract_tests": api_test_count,
            "telemetry_contract_tests": telemetry_test_count,
            "web_render_tests": web_test_count,
            "web_tiff_preview_tests": web_preview_test_count,
            "tiff_preview_decoder": "tiff@7.1.3",
            "browser_package": web_package.get("name"),
            "tailwind_css": True,
            "private_frontend_preview": True,
            "public_frontend_url": "https://terraclass-land-use-classification.vercel.app",
            "production_api_url": cloud_run_deployment.get("cloud_run", {}).get("service_url"),
            "production_api_deployed": cloud_run_deployment.get("claim_boundary", {}).get(
                "cloud_run_api_deployed"
            ),
            "integrated_deployment_claimed": cloud_run_deployment.get("claim_boundary", {}).get(
                "integrated_public_classifier_deployed"
            ),
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
