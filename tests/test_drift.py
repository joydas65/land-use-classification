import json
from pathlib import Path

import pytest

from terraclass.drift import (
    CONFIDENCE_BUCKETS,
    PROHIBITED_REVIEW_FIELDS,
    REVIEW_RECORD_FIELDS,
    DriftAnalysisConfig,
    build_report,
    compare_prediction_windows,
    jensen_shannon_divergence,
    load_drift_config,
    load_json_records,
    profile_prediction_window,
    summarize_human_reviews,
    validate_prediction_event,
)
from terraclass.telemetry import prediction_observation

CLASS_NAMES = ("agricultural", "airplane", "baseballdiamond", "beach", "buildings")


@pytest.fixture
def drift_config() -> DriftAnalysisConfig:
    return DriftAnalysisConfig(
        model_id="terraclass-resnet18-group-aware",
        model_version="1.0.0",
        class_names=CLASS_NAMES,
        minimum_window_predictions=5,
        minimum_reviewed_predictions=5,
        class_js_divergence=0.1,
        confidence_js_divergence=0.1,
        low_confidence_rate_increase=0.1,
        latency_p95_ratio=2.0,
    )


def prediction_event(
    index: int,
    *,
    class_name: str = "agricultural",
    confidence: float = 0.99,
    latency_ms: float = 20.0,
) -> dict[str, object]:
    return prediction_observation(
        request_id=f"request-{index}",
        model_id="terraclass-resnet18-group-aware",
        model_version="1.0.0",
        predicted_class=class_name,
        confidence=confidence,
        inference_latency_ms=latency_ms,
        image_width=256,
        image_height=256,
        payload_bytes=65_536,
        content_type="image/tiff",
    )


def review_record(index: int, predicted: str, reviewed: str) -> dict[str, object]:
    return {
        "event": "prediction_review",
        "schema_version": 1,
        "request_id": f"request-{index}",
        "model_id": "terraclass-resnet18-group-aware",
        "model_version": "1.0.0",
        "predicted_class": predicted,
        "reviewed_class": reviewed,
        "review_source": "manual_review",
        "reviewed_at_utc": f"2026-07-18T00:00:0{index}Z",
    }


def test_versioned_config_keeps_candidate_signals_and_review_privacy(project_root: Path) -> None:
    path = project_root / "configs/monitoring/drift_analysis_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = load_drift_config(path)

    assert config.class_names == CLASS_NAMES
    assert config.minimum_window_predictions == 100
    assert config.minimum_reviewed_predictions == 100
    assert raw["candidate_signals"]["status"] == "engineering_defaults_not_validated_thresholds"
    assert tuple(raw["human_review"]["allowlisted_fields"]) == REVIEW_RECORD_FIELDS
    assert set(raw["human_review"]["prohibited_fields"]) == PROHIBITED_REVIEW_FIELDS
    assert raw["claim_boundary"]["drift_detector_validated"] is False
    assert raw["claim_boundary"]["production_accuracy_established"] is False


def test_prediction_profile_accepts_cloud_logging_entries_without_retaining_request_ids(
    drift_config: DriftAnalysisConfig,
) -> None:
    records = [
        {
            "timestamp": f"2026-07-18T00:00:0{index}Z",
            "jsonPayload": prediction_event(
                index,
                class_name=CLASS_NAMES[index],
                confidence=(0.45, 0.6, 0.8, 0.9, 0.99)[index],
                latency_ms=10.0 + index,
            ),
        }
        for index in range(5)
    ]

    profile = profile_prediction_window(records, drift_config)

    assert profile["window"] == {
        "prediction_count": 5,
        "minimum_required": 5,
        "minimum_met": True,
        "first_timestamp_utc": "2026-07-18T00:00:00Z",
        "last_timestamp_utc": "2026-07-18T00:00:04Z",
    }
    assert profile["class_counts"] == {class_name: 1 for class_name in CLASS_NAMES}
    assert profile["confidence_bucket_counts"] == {bucket: 1 for bucket in CONFIDENCE_BUCKETS}
    assert profile["low_confidence_rate"] == 0.4
    assert profile["inference_latency_ms"]["p95"] == pytest.approx(13.8)
    assert profile["privacy"]["request_ids_retained"] is False
    assert "request-0" not in json.dumps(profile)


def test_prediction_validation_rejects_contract_or_privacy_drift(
    drift_config: DriftAnalysisConfig,
) -> None:
    event = prediction_event(1)
    event["filename"] = "private-scene.tif"
    with pytest.raises(ValueError, match="fields differ"):
        validate_prediction_event(event, drift_config)

    event = prediction_event(2, confidence=0.8)
    event["confidence_bucket"] = "0_95_to_1_00"
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction_event(event, drift_config)


def test_jensen_shannon_and_candidate_comparison_detect_large_distribution_change(
    drift_config: DriftAnalysisConfig,
) -> None:
    reference_records = [
        prediction_event(index, class_name="agricultural", confidence=0.99, latency_ms=20)
        for index in range(5)
    ]
    current_records = [
        prediction_event(index + 10, class_name="beach", confidence=0.4, latency_ms=60)
        for index in range(5)
    ]
    reference = profile_prediction_window(reference_records, drift_config)
    current = profile_prediction_window(current_records, drift_config)

    assert jensen_shannon_divergence(
        reference["class_distribution"],
        current["class_distribution"],
        CLASS_NAMES,
    ) == pytest.approx(1.0)
    comparison = compare_prediction_windows(reference, current, drift_config)
    assert comparison["status"] == "comparison_complete_not_validated"
    assert comparison["signals"]["class_js_divergence"]["exceeded"] is True
    assert comparison["signals"]["confidence_js_divergence"]["exceeded"] is True
    assert comparison["signals"]["low_confidence_rate_increase"]["exceeded"] is True
    assert comparison["signals"]["latency_p95_ratio"]["exceeded"] is True
    assert comparison["candidate_signal_exceeded"] is True
    assert "do not prove semantic drift" in comparison["claim_boundary"]


def test_comparison_refuses_drift_conclusion_below_sample_floor(
    drift_config: DriftAnalysisConfig,
) -> None:
    profile = profile_prediction_window([prediction_event(1)], drift_config)
    comparison = compare_prediction_windows(profile, profile, drift_config)

    assert comparison["status"] == "insufficient_data"
    assert comparison["candidate_signal_exceeded"] is None
    assert "No drift conclusion" in comparison["claim_boundary"]


def test_human_review_summary_reports_scoped_accuracy_and_macro_f1(
    drift_config: DriftAnalysisConfig,
) -> None:
    reviews = [
        review_record(index, class_name, class_name) for index, class_name in enumerate(CLASS_NAMES)
    ]
    summary = summarize_human_reviews(reviews, drift_config)

    assert summary["minimum_met"] is True
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["privacy"] == {
        "request_ids_retained_in_summary": False,
        "reviewer_identity_collected": False,
        "images_or_filenames_collected": False,
    }
    assert summary["claim_boundary"] == "Metrics describe only this reviewed sample."
    assert "request-1" not in json.dumps(summary)


def test_report_builder_handles_jsonl_and_preserves_insufficient_data_boundary(
    tmp_path: Path,
    project_root: Path,
) -> None:
    current_path = tmp_path / "current.jsonl"
    current_path.write_text(
        "\n".join(json.dumps(prediction_event(index)) for index in range(2)) + "\n",
        encoding="utf-8",
    )

    assert len(load_json_records(current_path)) == 2
    report = build_report(
        current_path=current_path,
        reference_path=current_path,
        config_path=project_root / "configs/monitoring/drift_analysis_v1.json",
        generated_at_utc="2026-07-18T00:00:00Z",
    )

    assert report["generated_at_utc"] == "2026-07-18T00:00:00Z"
    assert report["comparison"]["status"] == "insufficient_data"
    assert report["claim_boundary"] == {
        "drift_detector_validated": False,
        "production_accuracy_established": False,
    }
