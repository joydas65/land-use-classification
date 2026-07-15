import itertools
import time

import pytest

from terraclass.load_testing import RequestMeasurement, percentile, run_load_level


def test_percentile_interpolates_and_validates_input() -> None:
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert percentile([10.0, 20.0], 95) == pytest.approx(19.5)
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_load_level_records_concurrency_latency_and_unique_request_ids() -> None:
    sequence = itertools.count()

    def fake_request(base_url: str, payload: bytes, content_type: str) -> RequestMeasurement:
        assert base_url == "http://api.test"
        assert payload == b"image"
        assert content_type == "image/png"
        time.sleep(0.002)
        return RequestMeasurement(
            request_id=f"request-{next(sequence)}",
            status_code=200,
            total_latency_ms=5.0,
            model_pipeline_latency_ms=2.0,
        )

    result = run_load_level(
        "http://api.test",
        b"image",
        "image/png",
        concurrency=2,
        request_count=6,
        request_function=fake_request,
    )
    assert result["concurrency"] == 2
    assert result["requests"] == 6
    assert result["failures"] == 0
    assert result["throughput_requests_per_second"] > 0
    assert result["total_latency_ms"]["p95"] == 5.0
    assert result["model_pipeline_latency_ms"]["p50"] == 2.0
