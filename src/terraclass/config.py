"""Typed configuration loading and validation for TerraClass experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source_url: str
    verified_mirror_url: str
    archive_sha256: str
    archive_size_bytes: int
    total_images: int
    total_classes: int
    all_classes: tuple[str, ...]
    selected_classes: tuple[str, ...]
    images_per_class: int
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float
    manifest_path: str
    manifest_sha256: str
    expected_counts: dict[str, int]


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    horizontal_flip_probability: float
    rotation_degrees: float
    color_jitter: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    optimizer: str
    scheduler: str
    scheduler_step_size: int
    scheduler_gamma: float


@dataclass(frozen=True)
class ObservedBaseline:
    parameter_count: int
    best_training_accuracy: float
    best_validation_accuracy: float
    test_accuracy: float
    test_macro_f1: float


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    seed: int
    dataset: DatasetConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    observed_baseline: ObservedBaseline

    @property
    def class_count(self) -> int:
        return len(self.dataset.selected_classes)

    @property
    def selected_image_count(self) -> int:
        return self.class_count * self.dataset.images_per_class

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"unsupported schema_version={self.schema_version}")
        if not self.experiment_name.strip():
            errors.append("experiment_name must not be blank")
        if self.seed < 0:
            errors.append("seed must be non-negative")
        if len(set(self.dataset.selected_classes)) != self.class_count:
            errors.append("selected_classes contains duplicates")
        if len(self.dataset.all_classes) != self.dataset.total_classes:
            errors.append("all_classes length differs from total_classes")
        if len(set(self.dataset.all_classes)) != self.dataset.total_classes:
            errors.append("all_classes contains duplicates")
        if tuple(sorted(self.dataset.all_classes)) != self.dataset.all_classes:
            errors.append("all_classes must be sorted for stable label assignment")
        if not set(self.dataset.selected_classes).issubset(self.dataset.all_classes):
            errors.append("selected_classes must be a subset of all_classes")
        if tuple(sorted(self.dataset.selected_classes)) != self.dataset.selected_classes:
            errors.append("selected_classes must be sorted for stable label assignment")
        if self.class_count < 2 or self.class_count > self.dataset.total_classes:
            errors.append("selected class count is outside the dataset bounds")
        if self.dataset.images_per_class <= 0:
            errors.append("images_per_class must be positive")
        if not self.dataset.verified_mirror_url.startswith("https://"):
            errors.append("verified_mirror_url must use HTTPS")
        if len(self.dataset.archive_sha256) != 64:
            errors.append("archive_sha256 must contain 64 hexadecimal characters")
        if self.dataset.archive_size_bytes <= 0:
            errors.append("archive_size_bytes must be positive")
        split_sum = self.split.train + self.split.validation + self.split.test
        if abs(split_sum - 1.0) > 1e-9:
            errors.append(f"split fractions must sum to 1.0, got {split_sum}")
        if min(self.split.train, self.split.validation, self.split.test) <= 0:
            errors.append("all split fractions must be positive")
        if not self.split.manifest_path:
            errors.append("manifest_path must not be blank")
        if len(self.split.manifest_sha256) != 64:
            errors.append("manifest_sha256 must contain 64 hexadecimal characters")
        if sum(self.split.expected_counts.values()) != self.selected_image_count:
            errors.append("expected split counts do not equal selected image count")
        if self.preprocessing.image_size <= 0:
            errors.append("image_size must be positive")
        if any(value <= 0 for value in self.preprocessing.normalization_std):
            errors.append("normalization standard deviations must be positive")
        if self.training.batch_size <= 0 or self.training.epochs <= 0:
            errors.append("batch_size and epochs must be positive")
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if not 0 <= self.observed_baseline.test_accuracy <= 1:
            errors.append("observed test accuracy must be in [0, 1]")
        if not 0 <= self.observed_baseline.test_macro_f1 <= 1:
            errors.append("observed macro F1 must be in [0, 1]")
        if errors:
            raise ValueError("Invalid experiment configuration: " + "; ".join(errors))


def _tuple3(values: list[float], field: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{field} must contain exactly three values")
    return (float(values[0]), float(values[1]), float(values[2]))


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    dataset = raw["dataset"]
    split = raw["split"]
    preprocessing = raw["preprocessing"]
    training = raw["training"]
    observed = raw["observed_baseline"]
    config = ExperimentConfig(
        schema_version=int(raw["schema_version"]),
        experiment_name=str(raw["experiment_name"]),
        seed=int(raw["seed"]),
        dataset=DatasetConfig(
            name=str(dataset["name"]),
            source_url=str(dataset["source_url"]),
            verified_mirror_url=str(dataset["verified_mirror_url"]),
            archive_sha256=str(dataset["archive_sha256"]),
            archive_size_bytes=int(dataset["archive_size_bytes"]),
            total_images=int(dataset["total_images"]),
            total_classes=int(dataset["total_classes"]),
            all_classes=tuple(dataset["all_classes"]),
            selected_classes=tuple(dataset["selected_classes"]),
            images_per_class=int(dataset["images_per_class"]),
            extensions=tuple(extension.lower() for extension in dataset["extensions"]),
        ),
        split=SplitConfig(
            train=float(split["train"]),
            validation=float(split["validation"]),
            test=float(split["test"]),
            manifest_path=str(split["manifest_path"]),
            manifest_sha256=str(split["manifest_sha256"]),
            expected_counts={key: int(value) for key, value in split["expected_counts"].items()},
        ),
        preprocessing=PreprocessingConfig(
            image_size=int(preprocessing["image_size"]),
            normalization_mean=_tuple3(preprocessing["normalization_mean"], "normalization_mean"),
            normalization_std=_tuple3(preprocessing["normalization_std"], "normalization_std"),
            horizontal_flip_probability=float(preprocessing["horizontal_flip_probability"]),
            rotation_degrees=float(preprocessing["rotation_degrees"]),
            color_jitter=float(preprocessing["color_jitter"]),
        ),
        training=TrainingConfig(
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            epochs=int(training["epochs"]),
            optimizer=str(training["optimizer"]),
            scheduler=str(training["scheduler"]),
            scheduler_step_size=int(training["scheduler_step_size"]),
            scheduler_gamma=float(training["scheduler_gamma"]),
        ),
        observed_baseline=ObservedBaseline(**observed),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))
