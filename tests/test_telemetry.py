import io
import json

import pytest

from terraclass.telemetry import (
    PREDICTION_OBSERVATION_FIELDS,
    PROHIBITED_PREDICTION_FIELDS,
    confidence_bucket,
    emit_structured_event,
    prediction_observation,
)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.0, "below_0_50"),
        (0.499, "below_0_50"),
        (0.5, "0_50_to_0_70"),
        (0.7, "0_70_to_0_85"),
        (0.85, "0_85_to_0_95"),
        (0.95, "0_95_to_1_00"),
        (1.0, "0_95_to_1_00"),
    ],
)
def test_confidence_bucket_boundaries(confidence: float, expected: str) -> None:
    assert confidence_bucket(confidence) == expected


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_confidence_bucket_rejects_invalid_values(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        confidence_bucket(confidence)


def test_prediction_observation_uses_only_the_privacy_allowlist() -> None:
    event = prediction_observation(
        request_id="request-1",
        model_id="terraclass-resnet18-group-aware",
        model_version="1.0.0",
        predicted_class="beach",
        confidence=0.91234567,
        inference_latency_ms=12.54321,
        image_width=256,
        image_height=256,
        payload_bytes=65_536,
        content_type="image/tiff",
    )

    assert tuple(event) == PREDICTION_OBSERVATION_FIELDS
    assert not PROHIBITED_PREDICTION_FIELDS.intersection(event)
    assert event["confidence"] == 0.912346
    assert event["confidence_bucket"] == "0_85_to_0_95"
    assert event["inference_latency_ms"] == 12.543


def test_structured_event_is_one_machine_parseable_json_line() -> None:
    stream = io.StringIO()
    emit_structured_event({"event": "test", "schema_version": 1}, stream=stream)
    rendered = stream.getvalue()

    assert rendered.endswith("\n")
    assert rendered.count("\n") == 1
    assert json.loads(rendered) == {"event": "test", "schema_version": 1}
