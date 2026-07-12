"""Integrity and near-duplicate audit for the UC Merced image archive."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from terraclass.config import ExperimentConfig
from terraclass.data import file_sha256


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a 64-bit difference hash suitable for duplicate screening."""
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def find_near_duplicates(
    entries: Sequence[tuple[str, str, int]], threshold: int = 4, limit: int = 100
) -> tuple[int, list[dict[str, Any]]]:
    """Compare `(path, sha256, perceptual_hash)` entries by Hamming distance."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    count = 0
    examples: list[dict[str, Any]] = []
    for left_index, (left_path, left_sha, left_hash) in enumerate(entries):
        for right_path, right_sha, right_hash in entries[left_index + 1 :]:
            if left_sha == right_sha:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= threshold:
                count += 1
                if len(examples) < limit:
                    examples.append(
                        {
                            "left": left_path,
                            "right": right_path,
                            "hamming_distance": distance,
                        }
                    )
    return count, examples


def audit_dataset(
    dataset_root: str | Path,
    config: ExperimentConfig,
    near_duplicate_threshold: int = 4,
) -> dict[str, Any]:
    root_input = Path(dataset_root)
    root = root_input.resolve()
    actual_classes = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    if actual_classes != config.dataset.all_classes:
        raise ValueError("Dataset class directories differ from the configured 21-class inventory")

    class_counts: dict[str, int] = {}
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    exact_hash_paths: dict[str, list[str]] = defaultdict(list)
    perceptual_entries: list[tuple[str, str, int]] = []
    total_bytes = 0
    supported = set(config.dataset.extensions)

    for class_name in config.dataset.all_classes:
        paths = sorted(
            path
            for path in (root / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in supported
        )
        class_counts[class_name] = len(paths)
        if len(paths) != config.dataset.images_per_class:
            raise ValueError(
                f"{class_name} has {len(paths)} images; expected {config.dataset.images_per_class}"
            )
        for path in paths:
            relative = path.relative_to(root).as_posix()
            total_bytes += path.stat().st_size
            content_hash = file_sha256(path)
            exact_hash_paths[content_hash].append(relative)
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                formats[str(image.format)] += 1
                perceptual_entries.append((relative, content_hash, difference_hash(image)))

    total_images = sum(class_counts.values())
    if total_images != config.dataset.total_images:
        raise ValueError(
            f"Dataset has {total_images} images; expected {config.dataset.total_images}"
        )
    exact_groups = [paths for paths in exact_hash_paths.values() if len(paths) > 1]
    near_count, near_examples = find_near_duplicates(
        perceptual_entries, threshold=near_duplicate_threshold
    )
    return {
        "schema_version": 1,
        "dataset_name": config.dataset.name,
        "dataset_root": root_input.as_posix(),
        "total_images": total_images,
        "total_classes": len(actual_classes),
        "class_counts": class_counts,
        "dimensions": dict(sorted(dimensions.items())),
        "color_modes": dict(sorted(modes.items())),
        "image_formats": dict(sorted(formats.items())),
        "total_image_bytes": total_bytes,
        "unique_content_hashes": len(exact_hash_paths),
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_groups": exact_groups[:100],
        "perceptual_hash": {
            "algorithm": "difference_hash_64bit",
            "hamming_threshold": near_duplicate_threshold,
            "candidate_pair_count": near_count,
            "candidate_examples": near_examples,
        },
    }
