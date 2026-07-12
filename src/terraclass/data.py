"""Dataset discovery, deterministic splitting, and manifest validation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from terraclass.config import ExperimentConfig


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    label: int


def discover_samples(dataset_root: str | Path, config: ExperimentConfig) -> list[Sample]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    missing_classes = set(config.dataset.selected_classes) - actual_directories
    if missing_classes:
        raise ValueError(f"Missing selected class directories: {sorted(missing_classes)}")

    samples: list[Sample] = []
    supported = set(config.dataset.extensions)
    for label, class_name in enumerate(config.dataset.selected_classes):
        class_dir = root / class_name
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        )
        if len(image_paths) != config.dataset.images_per_class:
            raise ValueError(
                f"{class_name} has {len(image_paths)} supported images; "
                f"expected {config.dataset.images_per_class}"
            )
        samples.extend(
            Sample(path=path, class_name=class_name, label=label) for path in image_paths
        )
    return samples


def stratified_split(
    samples: Sequence[Sample], config: ExperimentConfig
) -> dict[str, list[Sample]]:
    labels = [sample.label for sample in samples]
    train_size = int(config.split.train * len(samples))
    validation_size = int(config.split.validation * len(samples))
    indices = list(range(len(samples)))
    train_indices, temporary_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=labels,
        random_state=config.seed,
    )
    temporary_labels = [labels[index] for index in temporary_indices]
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        train_size=validation_size,
        stratify=temporary_labels,
        random_state=config.seed,
    )
    splits = {
        "train": [samples[index] for index in train_indices],
        "validation": [samples[index] for index in validation_indices],
        "test": [samples[index] for index in test_indices],
    }
    validate_splits(splits, config)
    return splits


def validate_splits(splits: dict[str, Sequence[Sample]], config: ExperimentConfig) -> None:
    expected_names = {"train", "validation", "test"}
    if set(splits) != expected_names:
        raise ValueError(f"Split keys must be {sorted(expected_names)}")
    all_paths: list[Path] = []
    for split_name, split_samples in splits.items():
        expected_count = config.split.expected_counts[split_name]
        if len(split_samples) != expected_count:
            raise ValueError(
                f"{split_name} has {len(split_samples)} samples; expected {expected_count}"
            )
        class_counts = Counter(sample.class_name for sample in split_samples)
        if set(class_counts) != set(config.dataset.selected_classes):
            raise ValueError(f"{split_name} does not contain every selected class")
        all_paths.extend(sample.path.resolve() for sample in split_samples)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("A file appears in more than one split")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    splits: dict[str, Sequence[Sample]],
    dataset_root: str | Path,
    include_hashes: bool = True,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root).resolve()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "relative_path", "class_name", "label", "sha256"]
        )
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            for sample in sorted(splits[split_name], key=lambda item: str(item.path)):
                resolved = sample.path.resolve()
                writer.writerow(
                    {
                        "split": split_name,
                        "relative_path": resolved.relative_to(root).as_posix(),
                        "class_name": sample.class_name,
                        "label": sample.label,
                        "sha256": file_sha256(resolved) if include_hashes else "",
                    }
                )


class ImagePathDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, samples: Sequence[Sample], transform: Callable[[Image.Image], torch.Tensor]
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label


def class_counts(samples: Iterable[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.class_name for sample in samples).items()))
