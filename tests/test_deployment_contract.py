import json
from pathlib import Path


def test_container_is_non_root_cpu_only_and_hash_fetches_model(project_root: Path) -> None:
    dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8")
    release = json.loads(
        (project_root / "configs/serving/model_release_v1.json").read_text(encoding="utf-8")
    )
    assert "FROM python:3.12-slim-bookworm AS runtime-base" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "pip==26.1.2" in dockerfile
    assert "USER terraclass" in dockerfile
    assert "TERRACLASS_FAIL_ON_MODEL_ERROR=true" in dockerfile
    assert "python scripts/fetch_serving_artifact.py" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert release["url"].startswith(
        "https://github.com/joydas65/land-use-classification/releases/download/"
    )

    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools>=83"]' in pyproject
    assert '"Pillow>=12.3"' in pyproject


def test_cloud_run_template_matches_api_capacity_and_frontend_origin(project_root: Path) -> None:
    template = (project_root / "deploy/cloud-run-service.template.yaml").read_text(encoding="utf-8")
    assert "containerConcurrency: 4" in template
    assert "memory: 2Gi" in template
    assert "serviceAccountName: terraclass-runtime@PROJECT_ID.iam.gserviceaccount.com" in template
    assert "TERRACLASS_MAX_CONCURRENT_INFERENCES" in template
    assert "https://terraclass-land-use-classification.vercel.app" in template
    assert "/api/v1/health/ready" in template
    assert "/api/v1/health/live" in template


def test_public_model_release_evidence_matches_distribution_contract(project_root: Path) -> None:
    release = json.loads(
        (project_root / "configs/serving/model_release_v1.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (project_root / "reports/model_release_verification_2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["repository"]["visibility"] == "public"
    assert evidence["release"]["tag"] == "model-v1.0.0"
    assert evidence["asset"] == {
        "name": release["asset_name"],
        "url": release["url"],
        "size_bytes": release["size_bytes"],
        "sha256": release["sha256"],
    }
    assert evidence["verification"] == {
        "method": "fresh unauthenticated HTTPS download",
        "public_access": True,
        "size_matches_contract": True,
        "sha256_matches_contract": True,
    }


def test_public_container_release_has_digest_sbom_and_provenance(project_root: Path) -> None:
    evidence = json.loads(
        (project_root / "reports/container_release_verification_2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    registry = evidence["registry"]
    index_digest = evidence["oci_index"]["digest"]
    assert evidence["workflow"]["conclusion"] == "success"
    assert evidence["workflow"]["run_id"] == 29_503_393_345
    assert registry["visibility"] == "public"
    assert registry["public_pull_verified"] is True
    assert registry["tags"] == {
        "api-v1.0.0": index_digest,
        "sha-3b5b074": index_digest,
    }
    assert evidence["image"]["platform"] == {"architecture": "amd64", "os": "linux"}
    assert evidence["attestation_manifest"]["subject_digest"] == evidence["image"]["digest"]
    assert {layer["predicate_type"] for layer in evidence["attestation_manifest"]["layers"]} == {
        "https://spdx.dev/Document",
        "https://slsa.dev/provenance/v1",
    }


def test_cloud_run_and_vercel_evidence_bind_the_released_image(project_root: Path) -> None:
    release = json.loads(
        (project_root / "reports/container_release_verification_2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    deployment = json.loads(
        (project_root / "reports/cloud_run_deployment_verification_2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    load_report = json.loads(
        (project_root / "reports/cloud_run_load_test_2026-07-16.json").read_text(encoding="utf-8")
    )
    cloud_run = deployment["cloud_run"]
    assert cloud_run["ready"] is True
    assert cloud_run["public_access"] is True
    assert cloud_run["traffic_percent"] == 100
    assert cloud_run["revision"] == "terraclass-api-v1-0-1"
    assert cloud_run["deployed_oci_index_digest"] == release["oci_index"]["digest"]
    assert cloud_run["resolved_linux_amd64_digest"] == release["image"]["digest"]
    assert cloud_run["runtime_identity"] == {
        "service_account": (
            "terraclass-runtime@land-use-classification-502614.iam.gserviceaccount.com"
        ),
        "project_roles": [],
        "replaced_default_identity_role": "roles/editor",
    }
    assert cloud_run["resources"]["min_instances"] == 0
    assert cloud_run["resources"]["max_instances"] == 3
    assert deployment["api_verification"]["prediction"]["predicted_class"] == "agricultural"
    assert (
        deployment["api_verification"]["cors_preflight"]["allowed_origin"]
        == (deployment["vercel"]["production_alias"])
    )
    assert sum(level["requests"] for level in load_report["levels"]) == 60
    assert sum(level["failures"] for level in load_report["levels"]) == 0
    assert deployment["claim_boundary"] == {
        "cloud_run_api_deployed": True,
        "integrated_public_classifier_deployed": True,
        "production_load_probe_completed": True,
        "production_slo_established": False,
        "scale_to_zero_cold_request_measured": False,
    }


def test_ci_covers_python_web_and_container_contracts(project_root: Path) -> None:
    ci = (project_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (project_root / ".github/workflows/container-release.yml").read_text(encoding="utf-8")
    for token in (
        "audit_consistency.py",
        "python -m pip_audit",
        "setuptools>=83",
        "pytest -q",
        "ruff check",
        "npm test",
        "npm run build:vercel",
        "target: runtime-base",
    ):
        assert token in ci
    for token in ("registry: ghcr.io", "target: production", "provenance: mode=max", "sbom: true"):
        assert token in release
