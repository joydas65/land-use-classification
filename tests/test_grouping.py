from pathlib import Path

import pytest

from terraclass.config import ExperimentConfig
from terraclass.data import Sample
from terraclass.grouping import (
    group_aware_stratified_split,
    grouping_summary,
    load_verified_groups,
    validate_group_isolation,
)


def _samples(root: Path, config: ExperimentConfig) -> list[Sample]:
    return [
        Sample(root / class_name / f"{class_name}{index:02d}.tif", class_name, label)
        for label, class_name in enumerate(config.dataset.selected_classes)
        for index in range(config.dataset.images_per_class)
    ]


def test_group_aware_split_is_deterministic_balanced_and_isolated(
    tmp_path: Path, baseline_config: ExperimentConfig
) -> None:
    samples = _samples(tmp_path, baseline_config)
    reviewed = {
        "airplane/airplane01.tif": "same_scene",
        "airplane/airplane02.tif": "same_scene",
    }
    first, first_groups = group_aware_stratified_split(samples, baseline_config, tmp_path, reviewed)
    second, second_groups = group_aware_stratified_split(
        samples, baseline_config, tmp_path, reviewed
    )
    assert first == second
    assert first_groups == second_groups
    assert {name: len(items) for name, items in first.items()} == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    validate_group_isolation(first, tmp_path, first_groups)


def test_grouping_rejects_unknown_reviewed_path(
    tmp_path: Path, baseline_config: ExperimentConfig
) -> None:
    with pytest.raises(ValueError, match="outside the selected dataset"):
        group_aware_stratified_split(
            _samples(tmp_path, baseline_config),
            baseline_config,
            tmp_path,
            {"forest/forest00.tif": "not_selected"},
        )


def test_load_verified_groups_and_summary(project_root: Path) -> None:
    reviewed = load_verified_groups(project_root / "data/grouping/verified_related_scenes.json")
    assert reviewed["airplane/airplane01.tif"] == "airplane_same_scene_01_02"
    summary = grouping_summary(reviewed)
    assert summary["multi_image_groups"] == 4
    assert summary["grouped_images"] == 14
