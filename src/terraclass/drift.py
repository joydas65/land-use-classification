"""Offline, privacy-preserving production review and drift analysis."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terraclass.load_testing import percentile
from terraclass.telemetry import (
    PREDICTION_OBSERVATION_FIELDS,
    PROHIBITED_PREDICTION_FIELDS,
    confidence_bucket,
)

CONFIDENCE_BUCKETS = (
    "below_0_50",
    "0_50_to_0_70",
    "0_70_to_0_85",
    "0_85_to_0_95",
    "0_95_to_1_00",
)
REVIEW_RECORD_FIELDS = (
    "event",
    "schema_version",
    "request_id",
    "model_id",
    "model_version",
    "predicted_class",
    "reviewed_class",
    "review_source",
    "reviewed_at_utc",
)
PROHIBITED_REVIEW_FIELDS = frozenset(
    {
        "filename",
        "image_bytes",
        "image_sha256",
        "remote_ip",
        "user_agent",
        "reviewer_name",
        "reviewer_email",
    }
)


@dataclass(frozen=True)
class DriftAnalysisConfig:
    model_id: str
    model_version: str
    class_names: tuple[str, ...]
    minimum_window_predictions: int
    minimum_reviewed_predictions: int
    class_js_divergence: float
    confidence_js_divergence: float
    low_confidence_rate_increase: float
    latency_p95_ratio: float

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_version:
            raise ValueError("model identity must be non-empty")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must contain at least two unique values")
        if self.minimum_window_predictions <= 0 or self.minimum_reviewed_predictions <= 0:
            raise ValueError("minimum sample sizes must be positive")
        for value in (
            self.class_js_divergence,
            self.confidence_js_divergence,
            self.low_confidence_rate_increase,
        ):
            if not 0 <= value <= 1:
                raise ValueError("distribution thresholds must be between 0 and 1")
        if self.latency_p95_ratio <= 1:
            raise ValueError("latency_p95_ratio must be greater than 1")


def load_drift_config(path: str | Path) -> DriftAnalysisConfig:
    """Load and validate the versioned analysis contract."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError("drift analysis schema_version must be 1")
    service = raw["service"]
    samples = raw["minimum_samples"]
    thresholds = raw["candidate_signals"]
    return DriftAnalysisConfig(
        model_id=service["model_id"],
        model_version=service["model_version"],
        class_names=tuple(service["class_names"]),
        minimum_window_predictions=int(samples["prediction_window"]),
        minimum_reviewed_predictions=int(samples["human_review"]),
        class_js_divergence=float(thresholds["class_js_divergence"]),
        confidence_js_divergence=float(thresholds["confidence_js_divergence"]),
        low_confidence_rate_increase=float(thresholds["low_confidence_rate_increase"]),
        latency_p95_ratio=float(thresholds["latency_p95_ratio"]),
    )


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array, one JSON object, or newline-delimited JSON records."""

    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content[0] in "[{":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            if not all(isinstance(item, dict) for item in parsed):
                raise ValueError("JSON array records must be objects")
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSON record on line {line_number} must be an object")
        records.append(record)
    return records


def _payload_and_timestamp(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
    payload = record.get("jsonPayload", record)
    if not isinstance(payload, Mapping):
        raise ValueError("jsonPayload must be an object")
    timestamp = record.get("timestamp")
    if timestamp is not None and not isinstance(timestamp, str):
        raise ValueError("Cloud Logging timestamp must be a string")
    return payload, timestamp


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def validate_prediction_event(
    record: Mapping[str, Any],
    config: DriftAnalysisConfig,
) -> tuple[dict[str, Any], str | None]:
    """Validate one raw event or Cloud Logging entry against the telemetry contract."""

    payload, timestamp = _payload_and_timestamp(record)
    if set(payload) != set(PREDICTION_OBSERVATION_FIELDS):
        unexpected = set(payload) - set(PREDICTION_OBSERVATION_FIELDS)
        missing = set(PREDICTION_OBSERVATION_FIELDS) - set(payload)
        raise ValueError(
            f"prediction event fields differ: unexpected={unexpected}, missing={missing}"
        )
    if PROHIBITED_PREDICTION_FIELDS.intersection(payload):
        raise ValueError("prediction event contains prohibited privacy fields")
    if payload["event"] != "prediction_observation" or payload["schema_version"] != 1:
        raise ValueError("prediction event identity is invalid")
    if payload["model_id"] != config.model_id or payload["model_version"] != config.model_version:
        raise ValueError("prediction event model identity differs from the analysis contract")
    if payload["predicted_class"] not in config.class_names:
        raise ValueError("predicted_class is outside the model class contract")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if payload["confidence_bucket"] != confidence_bucket(confidence):
        raise ValueError("confidence_bucket does not match confidence")
    latency = _finite_number(payload["inference_latency_ms"], "inference_latency_ms")
    if latency <= 0:
        raise ValueError("inference_latency_ms must be positive")
    for field in ("image_width", "image_height", "payload_bytes"):
        if (
            isinstance(payload[field], bool)
            or not isinstance(payload[field], int)
            or payload[field] <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")
    for field in ("request_id", "model_id", "model_version", "content_type"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string")
    return dict(payload), timestamp


def _distribution(counts: Counter[str], categories: Sequence[str], total: int) -> dict[str, float]:
    return {category: counts[category] / total for category in categories}


def profile_prediction_window(
    records: Iterable[Mapping[str, Any]],
    config: DriftAnalysisConfig,
) -> dict[str, Any]:
    """Aggregate a strict prediction window without retaining request-level records."""

    events: list[dict[str, Any]] = []
    timestamps: list[str] = []
    request_ids: set[str] = set()
    for record in records:
        event, timestamp = validate_prediction_event(record, config)
        if event["request_id"] in request_ids:
            raise ValueError("prediction window contains a duplicate request_id")
        request_ids.add(event["request_id"])
        events.append(event)
        if timestamp:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timestamps.append(timestamp)
    if not events:
        raise ValueError("prediction window must contain at least one event")

    class_counts = Counter(str(event["predicted_class"]) for event in events)
    bucket_counts = Counter(str(event["confidence_bucket"]) for event in events)
    latencies = [float(event["inference_latency_ms"]) for event in events]
    low_confidence = sum(float(event["confidence"]) < 0.70 for event in events)
    total = len(events)
    return {
        "schema_version": 1,
        "model": {"model_id": config.model_id, "model_version": config.model_version},
        "window": {
            "prediction_count": total,
            "minimum_required": config.minimum_window_predictions,
            "minimum_met": total >= config.minimum_window_predictions,
            "first_timestamp_utc": min(timestamps) if timestamps else None,
            "last_timestamp_utc": max(timestamps) if timestamps else None,
        },
        "class_counts": {name: class_counts[name] for name in config.class_names},
        "class_distribution": _distribution(class_counts, config.class_names, total),
        "confidence_bucket_counts": {
            bucket: bucket_counts[bucket] for bucket in CONFIDENCE_BUCKETS
        },
        "confidence_bucket_distribution": _distribution(
            bucket_counts,
            CONFIDENCE_BUCKETS,
            total,
        ),
        "low_confidence_rate": low_confidence / total,
        "inference_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "maximum": max(latencies),
        },
        "privacy": {
            "request_ids_retained": False,
            "request_level_events_retained": False,
            "images_or_identifiers_retained": False,
        },
    }


def jensen_shannon_divergence(
    reference: Mapping[str, float],
    current: Mapping[str, float],
    categories: Sequence[str],
) -> float:
    """Return base-2 Jensen–Shannon divergence, bounded between zero and one."""

    reference_values = [float(reference.get(category, 0)) for category in categories]
    current_values = [float(current.get(category, 0)) for category in categories]
    if any(value < 0 for value in reference_values + current_values):
        raise ValueError("distribution values must be non-negative")
    if not math.isclose(sum(reference_values), 1.0, abs_tol=1e-9) or not math.isclose(
        sum(current_values), 1.0, abs_tol=1e-9
    ):
        raise ValueError("each distribution must sum to 1")
    midpoint = [
        (left + right) / 2 for left, right in zip(reference_values, current_values, strict=True)
    ]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0
        )

    return (divergence(reference_values) + divergence(current_values)) / 2


def compare_prediction_windows(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    config: DriftAnalysisConfig,
) -> dict[str, Any]:
    """Compare aggregate windows using candidate, explicitly unvalidated signals."""

    for name, profile in (("reference", reference), ("current", current)):
        model = profile.get("model", {})
        if model != {"model_id": config.model_id, "model_version": config.model_version}:
            raise ValueError(f"{name} profile model identity differs from the analysis contract")
    reference_count = int(reference["window"]["prediction_count"])
    current_count = int(current["window"]["prediction_count"])
    minimum_met = min(reference_count, current_count) >= config.minimum_window_predictions
    if not minimum_met:
        return {
            "status": "insufficient_data",
            "reference_predictions": reference_count,
            "current_predictions": current_count,
            "minimum_required_per_window": config.minimum_window_predictions,
            "candidate_signal_exceeded": None,
            "claim_boundary": "No drift conclusion is permitted below the minimum sample floor.",
        }

    class_js = jensen_shannon_divergence(
        reference["class_distribution"],
        current["class_distribution"],
        config.class_names,
    )
    confidence_js = jensen_shannon_divergence(
        reference["confidence_bucket_distribution"],
        current["confidence_bucket_distribution"],
        CONFIDENCE_BUCKETS,
    )
    low_confidence_increase = float(current["low_confidence_rate"]) - float(
        reference["low_confidence_rate"]
    )
    reference_p95 = float(reference["inference_latency_ms"]["p95"])
    current_p95 = float(current["inference_latency_ms"]["p95"])
    latency_ratio = current_p95 / reference_p95
    signals = {
        "class_js_divergence": {
            "value": class_js,
            "candidate_threshold": config.class_js_divergence,
            "exceeded": class_js > config.class_js_divergence,
        },
        "confidence_js_divergence": {
            "value": confidence_js,
            "candidate_threshold": config.confidence_js_divergence,
            "exceeded": confidence_js > config.confidence_js_divergence,
        },
        "low_confidence_rate_increase": {
            "value": low_confidence_increase,
            "candidate_threshold": config.low_confidence_rate_increase,
            "exceeded": low_confidence_increase > config.low_confidence_rate_increase,
        },
        "latency_p95_ratio": {
            "value": latency_ratio,
            "candidate_threshold": config.latency_p95_ratio,
            "exceeded": latency_ratio > config.latency_p95_ratio,
        },
    }
    return {
        "status": "comparison_complete_not_validated",
        "reference_predictions": reference_count,
        "current_predictions": current_count,
        "minimum_required_per_window": config.minimum_window_predictions,
        "signals": signals,
        "candidate_signal_exceeded": any(signal["exceeded"] for signal in signals.values()),
        "claim_boundary": (
            "Candidate signals require human review; they do not prove semantic drift or "
            "real-world accuracy change."
        ),
    }


def validate_review_record(
    record: Mapping[str, Any],
    config: DriftAnalysisConfig,
) -> dict[str, Any]:
    """Validate an owner-controlled human review record with no reviewer identity."""

    if set(record) != set(REVIEW_RECORD_FIELDS):
        raise ValueError("review record fields differ from the privacy contract")
    if PROHIBITED_REVIEW_FIELDS.intersection(record):
        raise ValueError("review record contains prohibited privacy fields")
    if record["event"] != "prediction_review" or record["schema_version"] != 1:
        raise ValueError("review record identity is invalid")
    if record["model_id"] != config.model_id or record["model_version"] != config.model_version:
        raise ValueError("review record model identity differs from the analysis contract")
    for field in ("predicted_class", "reviewed_class"):
        if record[field] not in config.class_names:
            raise ValueError(f"{field} is outside the model class contract")
    if record["review_source"] not in {"domain_expert", "dataset_label", "manual_review"}:
        raise ValueError("review_source is not supported")
    if not isinstance(record["request_id"], str) or not record["request_id"]:
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(record["reviewed_at_utc"], str):
        raise ValueError("reviewed_at_utc must be a string")
    datetime.fromisoformat(record["reviewed_at_utc"].replace("Z", "+00:00"))
    return dict(record)


def summarize_human_reviews(
    records: Iterable[Mapping[str, Any]],
    config: DriftAnalysisConfig,
) -> dict[str, Any]:
    """Calculate accuracy and macro-F1 for a private, owner-reviewed sample."""

    reviews: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for raw in records:
        review = validate_review_record(raw, config)
        if review["request_id"] in request_ids:
            raise ValueError("review sample contains a duplicate request_id")
        request_ids.add(review["request_id"])
        reviews.append(review)
    if not reviews:
        raise ValueError("review sample must contain at least one record")

    correct = sum(review["predicted_class"] == review["reviewed_class"] for review in reviews)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_name in config.class_names:
        true_positive = sum(
            review["predicted_class"] == class_name and review["reviewed_class"] == class_name
            for review in reviews
        )
        false_positive = sum(
            review["predicted_class"] == class_name and review["reviewed_class"] != class_name
            for review in reviews
        )
        false_negative = sum(
            review["predicted_class"] != class_name and review["reviewed_class"] == class_name
            for review in reviews
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        f1_values.append(f1)
        per_class[class_name] = {
            "support": sum(review["reviewed_class"] == class_name for review in reviews),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": f1,
        }
    review_count = len(reviews)
    minimum_met = review_count >= config.minimum_reviewed_predictions
    return {
        "schema_version": 1,
        "model": {"model_id": config.model_id, "model_version": config.model_version},
        "review_count": review_count,
        "minimum_required": config.minimum_reviewed_predictions,
        "minimum_met": minimum_met,
        "accuracy": correct / review_count,
        "macro_f1": statistics.fmean(f1_values),
        "per_class": per_class,
        "privacy": {
            "request_ids_retained_in_summary": False,
            "reviewer_identity_collected": False,
            "images_or_filenames_collected": False,
        },
        "claim_boundary": (
            "Metrics describe only this reviewed sample."
            if minimum_met
            else "Sample is below the review floor and must not be used as a production claim."
        ),
    }


def build_report(
    *,
    current_path: Path,
    config_path: Path,
    reference_path: Path | None = None,
    reviews_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the versioned aggregate report written by the command-line workflow."""

    config = load_drift_config(config_path)
    current_profile = profile_prediction_window(load_json_records(current_path), config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "current_profile": current_profile,
        "comparison": None,
        "human_review": None,
        "claim_boundary": {
            "drift_detector_validated": False,
            "production_accuracy_established": False,
        },
    }
    if reference_path is not None:
        reference_profile = profile_prediction_window(load_json_records(reference_path), config)
        report["reference_profile"] = reference_profile
        report["comparison"] = compare_prediction_windows(
            reference_profile,
            current_profile,
            config,
        )
    if reviews_path is not None:
        report["human_review"] = summarize_human_reviews(
            load_json_records(reviews_path),
            config,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build privacy-safe production profiles and compare candidate drift signals."
    )
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/monitoring/drift_analysis_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        current_path=args.current,
        config_path=args.config,
        reference_path=args.reference,
        reviews_path=args.reviews,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
