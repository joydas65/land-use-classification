from dataclasses import replace

import pytest

from terraclass.config import ExperimentConfig


def test_baseline_configuration_contract(baseline_config: ExperimentConfig) -> None:
    assert baseline_config.class_count == 5
    assert baseline_config.selected_image_count == 500
    assert baseline_config.dataset.total_classes == 21
    assert len(baseline_config.dataset.all_classes) == 21
    assert baseline_config.dataset.verified_mirror_url.startswith("https://")
    assert baseline_config.dataset.archive_size_bytes == 332_468_434
    assert baseline_config.dataset.archive_sha256 == (
        "06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb"
    )
    assert baseline_config.split.expected_counts == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    assert baseline_config.split.manifest_path == "data/manifests/baseline_5class_seed42.csv"
    assert baseline_config.split.manifest_sha256 == (
        "73d19e048e742fdf616cbbc1f037efa009ea329ec600acef329f2a5bc7df87ea"
    )
    assert baseline_config.training.batch_size == 32
    assert baseline_config.training.epochs == 10
    assert baseline_config.observed_baseline.test_accuracy == pytest.approx(0.7467)
    assert baseline_config.observed_baseline.test_macro_f1 == pytest.approx(0.733)


def test_configuration_rejects_invalid_split(baseline_config: ExperimentConfig) -> None:
    invalid = replace(
        baseline_config,
        split=replace(baseline_config.split, test=0.20),
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        invalid.validate()


def test_configuration_rejects_unstable_class_order(baseline_config: ExperimentConfig) -> None:
    invalid = replace(
        baseline_config,
        dataset=replace(
            baseline_config.dataset,
            selected_classes=tuple(reversed(baseline_config.dataset.selected_classes)),
        ),
    )
    with pytest.raises(ValueError, match="sorted"):
        invalid.validate()
