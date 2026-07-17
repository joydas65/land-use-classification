import numpy as np
import pytest
import torch
from torch import nn

from terraclass.data import Sample
from terraclass.model_quality import (
    calibration_metrics,
    compute_gradcam,
    deterministic_explainability_samples,
    fit_temperature,
    load_model_quality_config,
    reliability_bins,
    selective_prediction_curve,
)


def example_logits() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [3.0, 0.5, -1.0],
                [0.1, 2.4, -0.5],
                [-0.2, 0.3, 2.1],
                [2.0, 0.2, -0.1],
            ]
        ),
        torch.tensor([0, 1, 2, 0]),
    )


def test_model_quality_config_requires_validation_fit_and_test_evaluation(project_root) -> None:
    config = load_model_quality_config(project_root / "configs/evaluation/model_quality_v1.json")
    assert config.calibration.fit_split == "validation"
    assert config.calibration.evaluation_split == "test"
    assert config.explainability.selection_policy == (
        "lexicographically_first_test_sample_per_class"
    )


def test_calibration_metrics_are_finite_and_preserve_classification() -> None:
    logits, labels = example_logits()
    metrics = calibration_metrics(logits, labels, bin_count=5)
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert 0 < metrics["negative_log_likelihood"] < 1
    assert 0 <= metrics["expected_calibration_error"] <= 1
    assert 0 <= metrics["mean_normalized_predictive_entropy"] <= 1
    assert sum(row["count"] for row in metrics["reliability_bins"]) == 4


def test_reliability_bins_include_probability_one_in_final_bin() -> None:
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    labels = torch.tensor([0, 1])
    bins = reliability_bins(logits, labels, bin_count=10)
    assert sum(row["count"] for row in bins) == 2
    assert bins[-1]["count"] == 2
    assert bins[-1]["mean_confidence"] == 1.0


def test_temperature_fit_uses_bounded_validation_nll_and_flags_boundary() -> None:
    logits, labels = example_logits()
    fitted = fit_temperature(logits, labels, bounds=(0.05, 10.0), iterations=64)
    assert fitted["validation_nll_after"] <= fitted["validation_nll_before"]
    assert fitted["temperature"] == pytest.approx(0.05)
    assert fitted["at_lower_bound"] is True
    assert fitted["at_optimization_bound"] is True


def test_selective_prediction_reports_coverage_and_empty_abstention() -> None:
    logits, labels = example_logits()
    rows = selective_prediction_curve(logits, labels, (0.0, 0.9, 1.0))
    assert rows[0] == {
        "confidence_threshold": 0.0,
        "retained_count": 4,
        "coverage": 1.0,
        "accuracy": 1.0,
        "selective_risk": 0.0,
    }
    assert rows[-1]["retained_count"] == 0
    assert rows[-1]["accuracy"] is None
    assert rows[-1]["selective_risk"] is None


def test_gradcam_returns_normalized_input_sized_heatmap() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 3),
    )
    image = torch.rand(1, 3, 16, 16)
    heatmap, predicted, probabilities = compute_gradcam(model, image, model[0])
    assert heatmap.shape == (16, 16)
    assert np.isfinite(heatmap).all()
    assert 0 <= heatmap.min() <= heatmap.max() <= 1
    assert 0 <= predicted < 3
    assert probabilities.sum() == pytest.approx(1.0)


def test_explainability_selection_is_one_lexicographic_sample_per_class(tmp_path) -> None:
    samples = [
        Sample(tmp_path / "b2.tif", "b", 1),
        Sample(tmp_path / "a2.tif", "a", 0),
        Sample(tmp_path / "a1.tif", "a", 0),
        Sample(tmp_path / "b1.tif", "b", 1),
    ]
    selected = deterministic_explainability_samples(samples, ("a", "b"))
    assert [sample.path.name for sample in selected] == ["a1.tif", "b1.tif"]
