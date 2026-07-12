import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from terraclass.config import ExperimentConfig
from terraclass.data import (
    Sample,
    discover_samples,
    stratified_split,
    validate_splits,
    write_manifest,
)


def _fake_baseline_samples(config: ExperimentConfig) -> list[Sample]:
    return [
        Sample(Path(f"/virtual/{class_name}/{index:03d}.tif"), class_name, label)
        for label, class_name in enumerate(config.dataset.selected_classes)
        for index in range(config.dataset.images_per_class)
    ]


def test_stratified_split_is_deterministic_balanced_and_disjoint(
    baseline_config: ExperimentConfig,
) -> None:
    samples = _fake_baseline_samples(baseline_config)
    first = stratified_split(samples, baseline_config)
    second = stratified_split(samples, baseline_config)
    assert {name: values for name, values in first.items()} == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    expected_per_class = {"train": 70, "validation": 15, "test": 15}
    for split_name, split_samples in first.items():
        assert set(Counter(sample.class_name for sample in split_samples).values()) == {
            expected_per_class[split_name]
        }


def test_split_validation_detects_overlap(baseline_config: ExperimentConfig) -> None:
    splits = stratified_split(_fake_baseline_samples(baseline_config), baseline_config)
    splits["test"][0] = splits["train"][0]
    with pytest.raises(ValueError, match="more than one split"):
        validate_splits(splits, baseline_config)


def test_discovery_filters_extensions_and_uses_stable_labels(
    tmp_path: Path, baseline_config: ExperimentConfig
) -> None:
    reduced = replace(
        baseline_config,
        dataset=replace(baseline_config.dataset, images_per_class=2),
    )
    for class_name in reduced.dataset.selected_classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        Image.new("RGB", (8, 8), "red").save(class_dir / "b.tif")
        Image.new("RGB", (8, 8), "blue").save(class_dir / "a.tiff")
        (class_dir / "ignored.txt").write_text("not an image", encoding="utf-8")
    samples = discover_samples(tmp_path, reduced)
    assert len(samples) == 10
    assert samples[0].class_name == "agricultural"
    assert samples[0].label == 0
    assert samples[0].path.name == "a.tiff"


def test_manifest_contains_relative_paths_and_hashes(
    tmp_path: Path, baseline_config: ExperimentConfig
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    paths = []
    for split_name in ("train", "validation", "test"):
        image_path = dataset_root / f"{split_name}.tif"
        Image.new("RGB", (4, 4), "green").save(image_path)
        paths.append((split_name, image_path))

    def make_sample(path: Path) -> Sample:
        return Sample(path, "agricultural", 0)

    splits = {name: [make_sample(path)] for name, path in paths}
    destination = tmp_path / "manifest.csv"
    write_manifest(destination, splits, dataset_root, include_hashes=True)
    with destination.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["train", "validation", "test"]
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
