"""Privacy-preserving structured telemetry for TerraClass inference."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, TextIO

PREDICTION_OBSERVATION_FIELDS = (
    "event",
    "schema_version",
    "request_id",
    "model_id",
    "model_version",
    "predicted_class",
    "confidence",
    "confidence_bucket",
    "inference_latency_ms",
    "image_width",
    "image_height",
    "payload_bytes",
    "content_type",
)

PROHIBITED_PREDICTION_FIELDS = frozenset(
    {
        "filename",
        "image_bytes",
        "image_sha256",
        "remote_ip",
        "user_agent",
    }
)


def confidence_bucket(confidence: float) -> str:
    """Map a finite probability to a stable, low-cardinality monitoring bucket."""

    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be finite and between 0 and 1")
    if confidence < 0.50:
        return "below_0_50"
    if confidence < 0.70:
        return "0_50_to_0_70"
    if confidence < 0.85:
        return "0_70_to_0_85"
    if confidence < 0.95:
        return "0_85_to_0_95"
    return "0_95_to_1_00"


def prediction_observation(
    *,
    request_id: str,
    model_id: str,
    model_version: str,
    predicted_class: str,
    confidence: float,
    inference_latency_ms: float,
    image_width: int,
    image_height: int,
    payload_bytes: int,
    content_type: str,
) -> dict[str, Any]:
    """Build the allowlisted event emitted after a successful prediction."""

    if not request_id or not model_id or not model_version or not predicted_class:
        raise ValueError("prediction identity fields must be non-empty")
    if not math.isfinite(inference_latency_ms) or inference_latency_ms <= 0:
        raise ValueError("inference_latency_ms must be positive and finite")
    if image_width <= 0 or image_height <= 0 or payload_bytes <= 0:
        raise ValueError("image dimensions and payload_bytes must be positive")
    if not content_type:
        raise ValueError("content_type must be non-empty")

    event = {
        "event": "prediction_observation",
        "schema_version": 1,
        "request_id": request_id,
        "model_id": model_id,
        "model_version": model_version,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "confidence_bucket": confidence_bucket(confidence),
        "inference_latency_ms": round(inference_latency_ms, 3),
        "image_width": image_width,
        "image_height": image_height,
        "payload_bytes": payload_bytes,
        "content_type": content_type,
    }
    if tuple(event) != PREDICTION_OBSERVATION_FIELDS:
        raise RuntimeError("prediction telemetry field contract changed")
    return event


def emit_structured_event(event: Mapping[str, Any], *, stream: TextIO | None = None) -> None:
    """Write one compact JSON object so Cloud Logging stores a structured payload."""

    rendered = json.dumps(dict(event), separators=(",", ":"), sort_keys=True)
    print(rendered, file=stream, flush=True)
