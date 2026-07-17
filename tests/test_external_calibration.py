import pytest
import torch

from terraclass.external_calibration import (
    bootstrap_temperature_interval,
    cross_validated_temperature_stability,
    evaluate_promotion_gates,
    load_external_calibration_config,
    ood_detection_metrics,
    select_aligned_filenames,
    source_class_from_filename,
)
from terraclass.model_quality import calibration_metrics, fit_temperature


def imperfect_logits() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    labels = torch.arange(5).repeat_interleave(40)
    logits = torch.randn(200, 5, generator=generator) * 0.7
    logits[torch.arange(200), labels] += 1.2
    logits[:30] *= 3.0
    return logits, labels


def test_external_config_preserves_license_mapping_and_test_isolation(project_root) -> None:
    config = load_external_calibration_config(
        project_root / "configs/evaluation/external_calibration_v1.json"
    )
    assert config["external_dataset"]["redistribution"]["commercial_use_permitted"] is False
    assert config["class_mapping"]["buildings"] == ["commercial_area"]
    assert config["sampling"]["calibration_source_split"] == "validation"
    assert config["sampling"]["evaluation_source_split"] == "test"
    assert config["sampling"]["aligned_samples_per_target_class"] == 100
    assert config["promotion_gates"]["automatic_production_promotion"] is False


def test_resisc45_filename_parser_is_strict() -> None:
    assert source_class_from_filename("baseball_diamond_042.jpg") == "baseball_diamond"
    with pytest.raises(ValueError, match="Invalid RESISC45"):
        source_class_from_filename("../airplane_001.jpg")
    with pytest.raises(ValueError, match="Invalid RESISC45"):
        source_class_from_filename("airplane.jpg")


def test_aligned_selection_is_balanced_deterministic_and_disjoint_by_source() -> None:
    mapping = {
        "agricultural": ["circular_farmland", "rectangular_farmland"],
        "airplane": ["airplane"],
    }
    filenames = [
        *(f"circular_farmland_{index:03}.jpg" for index in range(20)),
        *(f"rectangular_farmland_{index:03}.jpg" for index in range(20)),
        *(f"airplane_{index:03}.jpg" for index in range(40)),
    ]
    first = select_aligned_filenames(
        filenames,
        mapping,
        per_target_class=20,
        seed=42,
    )
    second = select_aligned_filenames(
        list(reversed(filenames)),
        mapping,
        per_target_class=20,
        seed=42,
    )
    assert first == second
    assert sum(target == "agricultural" for _, target, _ in first) == 20
    assert sum(target == "airplane" for _, target, _ in first) == 20
    assert sum(name.startswith("circular_farmland") for name, _, _ in first) == 10


def test_bootstrap_temperature_interval_is_reproducible() -> None:
    logits, labels = imperfect_logits()
    first = bootstrap_temperature_interval(
        logits,
        labels,
        bounds=(0.05, 10.0),
        iterations=32,
        replicates=100,
        confidence_level=0.95,
        seed=42,
    )
    second = bootstrap_temperature_interval(
        logits,
        labels,
        bounds=(0.05, 10.0),
        iterations=32,
        replicates=100,
        confidence_level=0.95,
        seed=42,
    )
    assert first == second
    assert first["temperature"]["lower"] < first["temperature"]["upper"]
    assert first["temperature"]["bound_hits"] == 0


def test_cross_validated_temperature_reports_holdout_behavior() -> None:
    logits, labels = imperfect_logits()
    result = cross_validated_temperature_stability(
        logits,
        labels,
        bounds=(0.05, 10.0),
        iterations=32,
        folds=5,
        bin_count=10,
        seed=42,
    )
    assert result["folds"] == 5
    assert len(result["results"]) == 5
    assert result["all_fits_interior"] is True
    assert result["temperature_coefficient_of_variation"] >= 0


def test_ood_metrics_detect_separated_confidence_scores() -> None:
    aligned = torch.tensor([[8.0, 0.0]] * 20 + [[0.0, 8.0]] * 20)
    ood = torch.zeros(40, 2)
    result = ood_detection_metrics(aligned, ood, temperature=1.0)
    assert result["maximum_softmax_probability"]["auroc"] == 1.0
    assert result["maximum_softmax_probability"]["fpr_at_95_tpr"] == 0.0
    assert result["negative_normalized_entropy"]["auroc"] == 1.0


def test_promotion_requires_statistics_and_clears_no_policy_blocker(project_root) -> None:
    config = load_external_calibration_config(
        project_root / "configs/evaluation/external_calibration_v1.json"
    )
    logits, labels = imperfect_logits()
    fit = fit_temperature(logits, labels, bounds=(0.05, 10.0), iterations=32)
    before = calibration_metrics(logits, labels, bin_count=10)
    after = calibration_metrics(logits / float(fit["temperature"]), labels, bin_count=10)
    bootstrap = {
        "temperature": {
            "relative_interval_width": 0.1,
            "bound_hits": 0,
        }
    }
    cross_validation = {
        "temperature_coefficient_of_variation": 0.1,
        "all_fits_interior": True,
        "all_holdout_nll_improved": True,
    }
    result = evaluate_promotion_gates(
        config=config,
        fit=fit,
        bootstrap=bootstrap,
        cross_validation=cross_validation,
        calibration_before=before,
        external_before=before,
        external_after=after,
        in_domain_before=before,
        in_domain_after=before,
    )
    assert result["production_promotion_approved"] is False
    assert all(result["policy_blockers"].values())
    assert result["policy_blockers"]["license_not_approved_for_commercial_production"] is True
