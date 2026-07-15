import pytest

from scripts.benchmark_inference import percentile, summarize


def test_percentile_uses_linear_interpolation() -> None:
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == pytest.approx(38.5)
    assert percentile(values, 100) == 40.0


def test_latency_summary_is_complete() -> None:
    summary = summarize([10.0, 20.0, 30.0])
    assert summary == {
        "mean": 20.0,
        "p50": 20.0,
        "p95": pytest.approx(29.0),
        "minimum": 10.0,
        "maximum": 30.0,
    }


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="percentage"):
        percentile([1.0], 101)
