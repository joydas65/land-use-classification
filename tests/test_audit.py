from pathlib import Path

from terraclass.audit import KNOWN_ISSUE_IDS, audit_project


def test_cross_artifact_consistency_audit_passes(project_root: Path) -> None:
    report = audit_project(project_root)
    assert report.ok, report.errors
    assert report.observed["test_accuracy"] == 0.7467
    assert report.observed["test_macro_f1"] == 0.733
    assert report.observed["parameter_count"] == 102_277
    assert report.observed["known_issue_ids"] == list(KNOWN_ISSUE_IDS)
    assert report.observed["serving_foundation"]["model_version"] == "1.0.0"
    assert report.observed["serving_foundation"]["request_latency_p50_ms"] > 0
    assert (
        report.observed["serving_foundation"]["request_latency_p95_ms"]
        >= (report.observed["serving_foundation"]["request_latency_p50_ms"])
    )
    assert report.observed["production_readiness"]["container_contract"] is True
    assert report.observed["production_readiness"]["ci_workflows"] is True
    assert report.observed["production_readiness"]["model_release_public_download_verified"] is True
    assert len(report.observed["production_readiness"]["model_release_target_commit"]) == 40
    assert report.observed["production_readiness"]["container_release_workflow_run"] == 29503393345
    assert report.observed["production_readiness"]["container_public_pull_verified"] is True
    assert report.observed["production_readiness"]["container_sbom_attested"] is True
    assert report.observed["production_readiness"]["container_provenance_attested"] is True
    assert report.observed["production_readiness"]["container_index_digest"].startswith("sha256:")
    assert report.observed["production_readiness"]["container_platform_digest"].startswith(
        "sha256:"
    )
    assert report.observed["production_readiness"]["local_http_warmup_requests"] == 5
    assert report.observed["production_readiness"]["local_http_measured_requests"] == 60
    assert report.observed["production_readiness"]["local_http_concurrency_levels"] == [1, 2, 4]
    assert report.observed["production_readiness"]["local_http_peak_throughput_rps"] > 0
    assert report.observed["production_readiness"]["local_http_concurrency_4_p95_ms"] > 0
    assert report.observed["production_readiness"]["production_api_deployed"] is True
    assert (
        report.observed["production_readiness"]["cloud_run_service_url"]
        == "https://terraclass-api-280836764570.asia-south1.run.app"
    )
    assert report.observed["production_readiness"]["cloud_run_revision"] == (
        "terraclass-api-v1-0-1"
    )
    assert report.observed["production_readiness"]["cloud_run_region"] == "asia-south1"
    assert (
        report.observed["production_readiness"]["cloud_run_rollout_container_healthy_seconds"]
        == 18.02
    )
    assert (
        report.observed["production_readiness"]["cloud_run_scale_to_zero_cold_request_measured"]
        is False
    )
    assert report.observed["production_readiness"]["cloud_run_http_warmup_requests"] == 5
    assert report.observed["production_readiness"]["cloud_run_http_measured_requests"] == 60
    assert report.observed["production_readiness"]["cloud_run_http_failures"] == 0
    assert report.observed["production_readiness"]["cloud_run_http_peak_throughput_rps"] > 0
    assert report.observed["production_readiness"]["cloud_run_http_concurrency_4_p95_ms"] > 0
    assert report.observed["production_readiness"]["production_slo_established"] is False
    assert "29457675941" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
        encoding="utf-8"
    )
    assert "29503393345" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
        encoding="utf-8"
    )
    assert report.observed["application_layer"] == {
        "api_routes": [
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/model",
            "/api/v1/predictions",
        ],
        "api_contract_tests": 6,
        "web_render_tests": 2,
        "browser_package": "terraclass-web",
        "tailwind_css": True,
        "private_frontend_preview": True,
        "public_frontend_url": "https://terraclass-land-use-classification.vercel.app",
        "production_api_url": "https://terraclass-api-280836764570.asia-south1.run.app",
        "production_api_deployed": True,
        "integrated_deployment_claimed": True,
    }
