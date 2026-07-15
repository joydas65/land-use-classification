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
    assert "TERRACLASS_MAX_CONCURRENT_INFERENCES" in template
    assert "https://terraclass-land-use-classification.vercel.app" in template
    assert "/api/v1/health/ready" in template
    assert "/api/v1/health/live" in template


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
