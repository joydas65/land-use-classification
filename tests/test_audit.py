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
    assert report.observed["model_quality"] == {
        "scheduled_date": "2026-07-19",
        "completed_early": True,
        "calibration_fit_split": "validation",
        "calibration_fit_samples": 75,
        "evaluation_split": "test",
        "evaluation_samples": 75,
        "original_test_ece": 0.017477123268026102,
        "original_test_nll": 0.01974693908636797,
        "fitted_temperature": 0.05,
        "fit_reached_bound": True,
        "calibration_deployment_approved": False,
        "gradcam_samples": 5,
        "model_quality_contract_tests": 7,
    }
    assert report.observed["external_calibration"] == {
        "evaluated_on": "2026-07-18",
        "calibration_samples": 500,
        "external_test_samples": 500,
        "ood_samples": 5457,
        "temperature": 2.697718011917623,
        "fit_reached_bound": False,
        "external_test_ece_before": 0.21998949330946,
        "external_test_ece_after": 0.06607848174526312,
        "statistical_gates_passed": False,
        "production_promotion_approved": False,
        "external_calibration_contract_tests": 7,
        "external_download_contract_tests": 3,
    }
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
        is True
    )
    assert (
        report.observed["production_readiness"]["cloud_run_scale_to_zero_client_total_ms"]
        == 11013.115
    )
    assert report.observed["production_readiness"]["cloud_run_http_warmup_requests"] == 5
    assert report.observed["production_readiness"]["cloud_run_http_measured_requests"] == 60
    assert report.observed["production_readiness"]["cloud_run_http_failures"] == 0
    assert report.observed["production_readiness"]["cloud_run_http_peak_throughput_rps"] > 0
    assert report.observed["production_readiness"]["cloud_run_http_concurrency_4_p95_ms"] > 0
    assert report.observed["production_readiness"]["production_slo_established"] is False
    assert report.observed["production_readiness"]["candidate_availability_target"] == 0.99
    assert report.observed["production_readiness"]["candidate_p95_latency_ms"] == 1000
    assert report.observed["production_readiness"]["drift_readiness_status"] == (
        "telemetry_ready_not_drift_validated"
    )
    assert report.observed["production_readiness"]["cloud_monitoring_policies_deployed"] is True
    assert report.observed["production_readiness"]["cloud_monitoring_notifications_routed"] is False
    assert (
        report.observed["production_readiness"]["current_container_release_workflow_run"]
        == 29528840225
    )
    assert report.observed["production_readiness"]["current_container_index_digest"] == (
        "sha256:aee708b1d979a331f8f4f71ad9988ab01e6b04bc1cf2fc4420ad535328a06e41"
    )
    assert report.observed["production_readiness"]["current_container_platform_digest"] == (
        "sha256:eeb2e416780bbad8b86fad302916857c388a6375c5b86486244e8dad7e6e6f75"
    )
    assert report.observed["production_readiness"]["current_cloud_run_revision"] == (
        "terraclass-api-v1-1-0"
    )
    assert report.observed["production_readiness"]["current_service_version"] == "1.1.0"
    assert (
        report.observed["production_readiness"]["production_prediction_telemetry_observed"] is True
    )
    assert report.observed["production_readiness"]["operations_dashboard_deployed"] is True
    assert report.observed["production_readiness"]["operations_dashboard_name"] == (
        "projects/280836764570/dashboards/0c996266-70c0-4ad0-adc0-3e919225a4e4"
    )
    assert report.observed["production_readiness"]["production_prediction_inventory_count"] == 1
    assert (
        report.observed["production_readiness"]["production_prediction_inventory_minimum_met"]
        is False
    )
    assert report.observed["production_readiness"]["production_human_review_count"] == 0
    assert report.observed["production_readiness"]["production_drift_comparison_performed"] is False
    assert report.observed["production_readiness"]["production_drift_detector_validated"] is False
    assert report.observed["production_readiness"]["production_accuracy_established"] is False
    assert "filename" not in report.observed["production_readiness"]["prediction_telemetry_fields"]
    assert "29457675941" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
        encoding="utf-8"
    )
    assert "29503393345" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
        encoding="utf-8"
    )
    assert "29528840225" in (project_root / "docs/PRODUCTION_INFERENCE.md").read_text(
        encoding="utf-8"
    )
    assert report.observed["application_layer"] == {
        "api_routes": [
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/model",
            "/api/v1/predictions",
        ],
        "api_contract_tests": 7,
        "telemetry_contract_tests": 4,
        "drift_contract_tests": 7,
        "web_render_tests": 2,
        "web_tiff_preview_tests": 3,
        "tiff_preview_decoder": "tiff@7.1.3",
        "browser_package": "terraclass-web",
        "tailwind_css": True,
        "private_frontend_preview": True,
        "public_frontend_url": "https://terraclass-land-use-classification.vercel.app",
        "production_api_url": "https://terraclass-api-280836764570.asia-south1.run.app",
        "production_api_deployed": True,
        "integrated_deployment_claimed": True,
    }
