import hashlib
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


def test_observability_contract_is_private_and_matches_alert_policies(project_root: Path) -> None:
    config = json.loads(
        (project_root / "configs/monitoring/observability_v1.json").read_text(encoding="utf-8")
    )
    telemetry = config["telemetry"]
    objectives = config["candidate_objectives"]
    assert config["service"] == {
        "project_id": "land-use-classification-502614",
        "region": "asia-south1",
        "service_name": "terraclass-api",
        "service_version": "1.1.0",
        "model_id": "terraclass-resnet18-group-aware",
        "model_version": "1.0.0",
    }
    assert telemetry["image_content_retained"] is False
    assert set(telemetry["prohibited_fields"]) == {
        "filename",
        "image_bytes",
        "image_sha256",
        "remote_ip",
        "user_agent",
    }
    assert not set(telemetry["prohibited_fields"]).intersection(telemetry["allowlisted_fields"])
    assert objectives["status"] == "defined_not_yet_established"
    assert objectives["availability"]["target_ratio"] == 0.99
    assert objectives["steady_state_request_latency"]["threshold_ms"] == 1000
    assert config["drift_readiness"]["status"] == "telemetry_ready_not_drift_validated"

    policy_paths = config["alert_policy_templates"]
    policies = [
        json.loads((project_root / path).read_text(encoding="utf-8")) for path in policy_paths
    ]
    assert len(policies) == 2
    assert all(policy["enabled"] is True for policy in policies)
    assert {policy["userLabels"]["objective"] for policy in policies} == {
        "availability",
        "steady-state-latency",
    }
    serialized = json.dumps(policies)
    assert "run.googleapis.com/request_count" in serialized
    assert "run.googleapis.com/request_latencies" in serialized
    assert "terraclass-api" in serialized


def test_scale_to_zero_report_proves_zero_instances_and_a_new_autoscaled_instance(
    project_root: Path,
) -> None:
    evidence = json.loads(
        (project_root / "reports/cloud_run_scale_to_zero_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    precondition = evidence["scale_to_zero_precondition"]
    probe = evidence["client_probe"]
    corroboration = evidence["cloud_run_corroboration"]

    assert evidence["measured_on"] == "2026-07-17"
    assert evidence["service"]["minimum_instances"] == 0
    assert precondition["active_instances"] == 0
    assert precondition["idle_instances"] == 0
    assert (
        precondition["request_gap_seconds"]
        > precondition["documented_possible_idle_retention_seconds"]
    )
    assert probe["http_status"] == 200
    assert probe["response"]["predicted_class"] == "agricultural"
    assert probe["curl_timings_ms"]["total"] == 11013.115
    assert "AUTOSCALING" in corroboration["instance_start_log"]["reason"]
    assert corroboration["instance_start_log"]["matches_request_instance"] is True
    assert evidence["claim_boundary"]["scale_from_zero_client_request_measured"] is True
    assert evidence["claim_boundary"]["availability_slo_established"] is False


def test_cloud_monitoring_deployment_matches_templates_without_notification_routing(
    project_root: Path,
) -> None:
    evidence = json.loads(
        (project_root / "reports/cloud_monitoring_deployment_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    policies = evidence["policies"]

    assert evidence["project_id"] == "land-use-classification-502614"
    assert evidence["readback_verified"] is True
    assert {policy["template"] for policy in policies} == {
        "deploy/monitoring/cloud-run-5xx-ratio.json",
        "deploy/monitoring/cloud-run-p95-latency.json",
    }
    assert all(policy["enabled"] is True for policy in policies)
    assert all(policy["notification_channels"] == [] for policy in policies)
    assert evidence["claim_boundary"]["alert_policies_deployed"] is True
    assert evidence["claim_boundary"]["notifications_routed"] is False
    assert evidence["claim_boundary"]["candidate_objectives_established_as_slo"] is False


def test_operations_dashboard_and_first_drift_inventory_preserve_claim_boundaries(
    project_root: Path,
) -> None:
    dashboard_path = project_root / "deploy/monitoring/terraclass-operations-dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    evidence = json.loads(
        (project_root / "reports/production_drift_readiness_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    profile = evidence["production_prediction_inventory"]["profile"]
    serialized_dashboard = json.dumps(dashboard)

    assert dashboard["displayName"] == "TerraClass Production Operations"
    assert len(dashboard["mosaicLayout"]["tiles"]) == 5
    assert "run.googleapis.com/request_count" in serialized_dashboard
    assert "run.googleapis.com/request_latencies" in serialized_dashboard
    assert "run.googleapis.com/container/instance_count" in serialized_dashboard
    assert "prediction_observation" in serialized_dashboard
    assert (
        evidence["dashboard"]["definition_sha256"]
        == hashlib.sha256(dashboard_path.read_bytes()).hexdigest()
    )
    assert evidence["dashboard"]["name"] == (
        "projects/280836764570/dashboards/0c996266-70c0-4ad0-adc0-3e919225a4e4"
    )
    assert evidence["dashboard"]["readback_verified"] is True
    assert evidence["production_prediction_inventory"]["raw_request_level_log_committed"] is False
    assert profile["window"]["prediction_count"] == 1
    assert profile["window"]["minimum_required"] == 100
    assert profile["window"]["minimum_met"] is False
    assert profile["privacy"]["request_ids_retained"] is False
    assert evidence["human_review"]["reviewed_predictions"] == 0
    assert evidence["notification_routing"]["configured"] is False
    assert evidence["claim_boundary"]["drift_comparison_performed"] is False
    assert evidence["claim_boundary"]["drift_detector_validated"] is False
    assert evidence["claim_boundary"]["production_accuracy_established"] is False


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


def test_observability_container_release_binds_source_tags_and_attestations(
    project_root: Path,
) -> None:
    evidence = json.loads(
        (project_root / "reports/container_release_verification_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    index_digest = "sha256:aee708b1d979a331f8f4f71ad9988ab01e6b04bc1cf2fc4420ad535328a06e41"
    image_digest = "sha256:eeb2e416780bbad8b86fad302916857c388a6375c5b86486244e8dad7e6e6f75"

    assert evidence["source"] == {
        "repository": "https://github.com/joydas65/land-use-classification",
        "commit": "e8a11cf4ba71db65d96479dd35cd7064072dedd4",
        "tag": "api-v1.1.0",
    }
    assert evidence["workflow"]["run_id"] == 29_528_840_225
    assert evidence["workflow"]["conclusion"] == "success"
    assert evidence["registry"]["anonymous_oci_pull_verified"] is True
    assert evidence["registry"]["tags"] == {
        "api-v1.1.0": index_digest,
        "sha-e8a11cf": index_digest,
    }
    assert evidence["oci_index"]["digest"] == index_digest
    assert evidence["image"] == {
        "digest": image_digest,
        "media_type": "application/vnd.oci.image.manifest.v1+json",
        "platform": {"architecture": "amd64", "os": "linux"},
    }
    assert evidence["attestation_manifest"]["subject_digest"] == image_digest
    assert {layer["predicate_type"] for layer in evidence["attestation_manifest"]["layers"]} == {
        "https://spdx.dev/Document",
        "https://slsa.dev/provenance/v1",
    }


def test_observability_deployment_binds_release_telemetry_and_browser(
    project_root: Path,
) -> None:
    release = json.loads(
        (project_root / "reports/container_release_verification_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    deployment = json.loads(
        (project_root / "reports/cloud_run_observability_deployment_2026-07-17.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (project_root / "configs/monitoring/observability_v1.json").read_text(encoding="utf-8")
    )
    cloud_run = deployment["cloud_run"]
    api = deployment["api_verification"]
    telemetry = deployment["structured_telemetry_verification"]

    assert cloud_run["revision"] == "terraclass-api-v1-1-0"
    assert cloud_run["ready"] is True
    assert cloud_run["traffic_percent"] == 100
    assert cloud_run["deployed_oci_index_digest"] == release["oci_index"]["digest"]
    assert cloud_run["resolved_linux_amd64_digest"] == release["image"]["digest"]
    assert cloud_run["runtime_identity"]["project_roles"] == []
    assert cloud_run["resources"]["min_instances"] == 0
    assert api["service_version"] == "1.1.0"
    assert api["model_version"] == "1.0.0"
    assert api["prediction"]["predicted_class"] == "agricultural"
    assert telemetry["payload_type"] == "jsonPayload"
    assert set(telemetry["event"]) == set(config["telemetry"]["allowlisted_fields"])
    assert telemetry["prohibited_fields_absent"] == config["telemetry"]["prohibited_fields"]
    assert telemetry["event"]["request_id"] == api["prediction"]["request_id"]
    assert deployment["vercel_browser"]["browser_model_status"] == "Model ready"
    assert deployment["vercel_browser"]["console_errors"] == 0
    assert deployment["vercel_browser"]["console_warnings"] == 0
    assert deployment["claim_boundary"] == {
        "service_v1_1_0_deployed": True,
        "privacy_allowlisted_prediction_telemetry_observed": True,
        "alert_policies_deployed": True,
        "notifications_routed": False,
        "candidate_objectives_established_as_slo": False,
        "drift_detector_validated": False,
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
