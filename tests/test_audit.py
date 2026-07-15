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
    assert report.observed["application_layer"] == {
        "api_routes": [
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/model",
            "/api/v1/predictions",
        ],
        "api_contract_tests": 5,
        "web_render_tests": 2,
        "browser_package": "terraclass-web",
        "tailwind_css": True,
        "private_frontend_preview": True,
        "public_frontend_url": "https://terraclass-land-use-classification.vercel.app",
        "integrated_deployment_claimed": False,
    }
