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
    group_by_relative_path: dict[str, str] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root).resolve()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "relative_path", "class_name", "label", "sha256"]
        if group_by_relative_path is not None:
            fieldnames.append("group_id")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            for sample in sorted(splits[split_name], key=lambda item: str(item.path)):
                resolved = sample.path.resolve()
                relative_path = resolved.relative_to(root).as_posix()
                row = {
                    "split": split_name,
                    "relative_path": relative_path,
                    "class_name": sample.class_name,
                    "label": sample.label,
                    "sha256": file_sha256(resolved) if include_hashes else "",
                }
                if group_by_relative_path is not None:
                    if relative_path not in group_by_relative_path:
                        raise ValueError(f"Missing group ID for {relative_path}")
                    row["group_id"] = group_by_relative_path[relative_path]
                writer.writerow(row)


def load_manifest(
    path: str | Path,
    dataset_root: str | Path,
    config: ExperimentConfig,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, list[Sample]], dict[str, str]]:
    """Load a manifest, verify provenance, and enforce optional group isolation."""
    root = Path(dataset_root).resolve()
    splits: dict[str, list[Sample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    path_to_group: dict[str, str] = {}
    group_splits: dict[str, set[str]] = {}
    expected_labels = {
        class_name: label for label, class_name in enumerate(config.dataset.selected_classes)
    }
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "relative_path", "class_name", "label", "sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest is missing required columns: {sorted(required)}")
        for row in reader:
            split_name = row["split"]
            if split_name not in splits:
                raise ValueError(f"Unknown manifest split: {split_name}")
            relative = Path(row["relative_path"])
            if relative.is_absolute():
                raise ValueError("Manifest paths must be relative")
            resolved = (root / relative).resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"Manifest path escapes dataset root: {relative}")
            if not resolved.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {resolved}")
            class_name = row["class_name"]
            if class_name not in expected_labels:
                raise ValueError(f"Unexpected class in manifest: {class_name}")
            label = int(row["label"])
            if label != expected_labels[class_name]:
                raise ValueError(f"Label mismatch for {relative}")
            if verify_hashes and file_sha256(resolved) != row["sha256"]:
                raise ValueError(f"Image hash mismatch for {relative}")
            relative_posix = relative.as_posix()
            group_id = row.get("group_id") or f"singleton::{relative_posix}"
            path_to_group[relative_posix] = group_id
            group_splits.setdefault(group_id, set()).add(split_name)
            splits[split_name].append(Sample(resolved, class_name, label))
    validate_splits(splits, config)
    crossing = {
        group_id: sorted(names) for group_id, names in group_splits.items() if len(names) > 1
    }
    if crossing:
        raise ValueError(f"Manifest groups cross split boundaries: {crossing}")
    return splits, path_to_group


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
