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
    assert report.observed["production_readiness"]["local_http_warmup_requests"] == 5
    assert report.observed["production_readiness"]["local_http_measured_requests"] == 60
    assert report.observed["production_readiness"]["local_http_concurrency_levels"] == [1, 2, 4]
    assert report.observed["production_readiness"]["local_http_peak_throughput_rps"] > 0
    assert report.observed["production_readiness"]["local_http_concurrency_4_p95_ms"] > 0
    assert report.observed["production_readiness"]["production_api_deployed"] is False
    assert "29455400219" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
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
        "production_api_deployed": False,
        "integrated_deployment_claimed": False,
    }
