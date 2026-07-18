import numpy as np
import pytest
import torch
from PIL import Image

from terraclass.robustness import (
    CORRUPTION_NAMES,
    apply_corruption,
    corruption_scenarios,
    deterministic_sample_seed,
    evaluate_candidate_selection,
    evaluate_promotion,
    load_robustness_config,
    summarize_scenarios,
    tta_variant,
)


def metric_row(accuracy: float, macro_f1: float) -> dict[str, float | int]:
    return {
        "sample_count": 10,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "negative_log_likelihood": 0.5,
        "multiclass_brier_score": 0.25,
        "expected_calibration_error": 0.1,
        "mean_confidence": 0.8,
    }


def scenario_rows(candidate_delta: float = 0.01) -> list[dict]:
    rows = [
        {
            "id": "clean",
            "corruption": None,
            "severity": 0,
            "parameter": None,
            "baseline": metric_row(1.0, 1.0),
            "candidate": metric_row(1.0, 1.0),
            "baseline_seconds": 0.1,
            "candidate_seconds": 0.4,
        }
    ]
    for corruption in CORRUPTION_NAMES:
        for severity in (1, 2, 3):
            baseline_f1 = 0.95 - severity * 0.1
            rows.append(
                {
                    "id": f"{corruption}:severity_{severity}",
                    "corruption": corruption,
                    "severity": severity,
                    "parameter": severity,
                    "baseline": metric_row(baseline_f1, baseline_f1),
                    "candidate": metric_row(
                        baseline_f1 + candidate_delta,
                        baseline_f1 + candidate_delta,
                    ),
                    "baseline_seconds": 0.1,
                    "candidate_seconds": 0.4,
                }
            )
    return rows


def test_robustness_config_separates_selection_test_and_production(project_root) -> None:
    config = load_robustness_config(project_root / "configs/evaluation/robustness_v1.json")
    assert config["protocol"]["selection_split"] == "validation"
    assert config["protocol"]["evaluation_split"] == "test"
    assert tuple(config["protocol"]["corruptions"]) == CORRUPTION_NAMES
    assert config["claim_boundary"]["resisc45_used"] is False
    assert config["promotion_gates"]["automatic_production_promotion"] is False


def test_sample_seed_is_reproducible_and_identity_sensitive() -> None:
    first = deterministic_sample_seed(42, "beach/beach01.tif", 2)
    assert first == deterministic_sample_seed(42, "beach/beach01.tif", 2)
    assert first != deterministic_sample_seed(42, "beach/beach02.tif", 2)
    assert first != deterministic_sample_seed(42, "beach/beach01.tif", 3)
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_sample_seed(42, "beach/beach01.tif", -1)


@pytest.mark.parametrize(
    ("name", "parameter"),
    (
        ("brightness_reduction", 0.65),
        ("contrast_reduction", 0.6),
        ("gaussian_blur", 1.6),
        ("gaussian_noise", 0.05),
        ("jpeg_compression", 40),
    ),
)
def test_corruptions_are_deterministic_rgb_and_preserve_size(
    name: str,
    parameter: float,
) -> None:
    pixels = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(10, 12, 3)
    image = Image.fromarray(pixels)
    first = apply_corruption(image, name, parameter, seed=123)
    second = apply_corruption(image, name, parameter, seed=123)
    assert first.mode == "RGB"
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.array_equal(np.asarray(image), pixels)


def test_corruption_validation_rejects_invalid_contracts() -> None:
    image = Image.new("RGB", (8, 8), "white")
    with pytest.raises(ValueError, match="Unsupported"):
        apply_corruption(image, "unknown", 1, seed=1)
    with pytest.raises(ValueError, match="brightness"):
        apply_corruption(image, "brightness_reduction", 0, seed=1)
    with pytest.raises(ValueError, match="JPEG"):
        apply_corruption(image, "jpeg_compression", 96, seed=1)


def test_dihedral_tta_variants_have_exact_tensor_semantics() -> None:
    images = torch.arange(12).reshape(1, 1, 3, 4)
    assert torch.equal(tta_variant(images, "identity"), images)
    assert torch.equal(
        tta_variant(images, "horizontal_flip"),
        torch.flip(images, dims=(-1,)),
    )
    assert torch.equal(
        tta_variant(images, "vertical_flip"),
        torch.flip(images, dims=(-2,)),
    )
    assert torch.equal(
        tta_variant(images, "rotate_180"),
        torch.flip(images, dims=(-2, -1)),
    )
    with pytest.raises(ValueError, match="Unsupported"):
        tta_variant(images, "rotate_90")


def test_scenario_matrix_has_clean_plus_fifteen_corruptions(project_root) -> None:
    config = load_robustness_config(project_root / "configs/evaluation/robustness_v1.json")
    scenarios = corruption_scenarios(config)
    assert len(scenarios) == 16
    assert scenarios[0]["id"] == "clean"
    assert {row["severity"] for row in scenarios[1:]} == {1, 2, 3}
    assert len({row["id"] for row in scenarios}) == 16


def test_summary_reports_average_worst_case_and_latency() -> None:
    summary = summarize_scenarios(scenario_rows())
    assert summary["scenario_count"] == 16
    assert summary["corrupted_scenario_count"] == 15
    assert summary["baseline"]["worst_condition"]["severity"] == 3
    assert summary["candidate_vs_baseline"]["mean_corruption_macro_f1_delta"] == pytest.approx(0.01)
    assert summary["candidate_vs_baseline"]["latency_multiplier"] == pytest.approx(4.0)


def test_candidate_selection_uses_validation_only(project_root) -> None:
    config = load_robustness_config(project_root / "configs/evaluation/robustness_v1.json")
    selected = evaluate_candidate_selection(
        summarize_scenarios(scenario_rows(candidate_delta=0.01)),
        config,
    )
    rejected = evaluate_candidate_selection(
        summarize_scenarios(scenario_rows(candidate_delta=0.001)),
        config,
    )
    assert selected["selected_for_final_test"] is True
    assert selected["test_metrics_consulted"] is False
    assert rejected["selected_for_final_test"] is False


def test_production_promotion_remains_blocked_for_synthetic_evidence(
    project_root,
) -> None:
    config = load_robustness_config(project_root / "configs/evaluation/robustness_v1.json")
    summary = summarize_scenarios(scenario_rows(candidate_delta=0.01))
    selection = evaluate_candidate_selection(summary, config)
    promotion = evaluate_promotion(
        selection=selection,
        test_summary=summary,
        config=config,
    )
    assert promotion["evidence_gates_passed"] is True
    assert promotion["production_promotion_approved"] is False
    assert all(promotion["policy_blockers"].values())
