"""Deterministic grouping and group-aware stratification utilities."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from terraclass.config import ExperimentConfig
from terraclass.data import Sample, validate_splits

SPLIT_ORDER = ("train", "validation", "test")


def load_verified_groups(path: str | Path) -> dict[str, str]:
    """Return a mapping from dataset-relative path to reviewed group ID."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported grouping schema")
    path_to_group: dict[str, str] = {}
    for group in raw.get("groups", []):
        group_id = str(group["group_id"])
        class_name = str(group["class_name"])
        members = group.get("members", [])
        if len(members) < 2:
            raise ValueError(f"Reviewed group {group_id} must contain at least two members")
        for relative_path in members:
            relative_path = str(relative_path)
            if relative_path.split("/", 1)[0] != class_name:
                raise ValueError(f"Group {group_id} contains a path from another class")
            if relative_path in path_to_group:
                raise ValueError(f"{relative_path} appears in more than one reviewed group")
            path_to_group[relative_path] = group_id
    return path_to_group


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def group_aware_stratified_split(
    samples: Sequence[Sample],
    config: ExperimentConfig,
    dataset_root: str | Path,
    reviewed_path_to_group: dict[str, str],
) -> tuple[dict[str, list[Sample]], dict[str, str]]:
    """Split by class while keeping reviewed groups entirely in one split."""
    root = Path(dataset_root).resolve()
    sample_by_relative_path = {
        sample.path.resolve().relative_to(root).as_posix(): sample for sample in samples
    }
    unknown_reviewed_paths = set(reviewed_path_to_group) - set(sample_by_relative_path)
    if unknown_reviewed_paths:
        raise ValueError(
            f"Reviewed grouping contains paths outside the selected dataset: "
            f"{sorted(unknown_reviewed_paths)}"
        )

    path_to_group: dict[str, str] = {}
    for relative_path, sample in sample_by_relative_path.items():
        path_to_group[relative_path] = reviewed_path_to_group.get(
            relative_path, f"singleton::{sample.class_name}::{sample.path.stem}"
        )

    by_class_and_group: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for relative_path, sample in sample_by_relative_path.items():
        by_class_and_group[sample.class_name][path_to_group[relative_path]].append(sample)

    targets = {
        split_name: config.split.expected_counts[split_name] // config.class_count
        for split_name in SPLIT_ORDER
    }
    splits: dict[str, list[Sample]] = {name: [] for name in SPLIT_ORDER}
    group_to_split: dict[str, str] = {}

    for class_name in config.dataset.selected_classes:
        groups = by_class_and_group[class_name]
        multi_groups = sorted(
            ((group_id, members) for group_id, members in groups.items() if len(members) > 1),
            key=lambda item: (-len(item[1]), item[0]),
        )
        single_groups = sorted(
            ((group_id, members) for group_id, members in groups.items() if len(members) == 1),
            key=lambda item: item[0],
        )
        random.Random(_stable_seed(config.seed, class_name)).shuffle(single_groups)
        remaining = dict(targets)

        for group_id, members in multi_groups:
            eligible = [name for name in SPLIT_ORDER if remaining[name] >= len(members)]
            if not eligible:
                raise ValueError(
                    f"Reviewed group {group_id} cannot fit the remaining class capacities"
                )
            chosen = max(
                eligible,
                key=lambda name: (
                    remaining[name] / targets[name],
                    -SPLIT_ORDER.index(name),
                ),
            )
            splits[chosen].extend(members)
            remaining[chosen] -= len(members)
            group_to_split[group_id] = chosen

        for group_id, members in single_groups:
            eligible = [name for name in SPLIT_ORDER if remaining[name] > 0]
            if not eligible:
                raise ValueError(f"No remaining split capacity for singleton {group_id}")
            chosen = max(
                eligible,
                key=lambda name: (
                    remaining[name] / targets[name],
                    -SPLIT_ORDER.index(name),
                ),
            )
            splits[chosen].extend(members)
            remaining[chosen] -= 1
            group_to_split[group_id] = chosen

        if any(remaining.values()):
            raise ValueError(f"Class {class_name} did not meet target counts: {remaining}")

    validate_splits(splits, config)
    return splits, path_to_group


def validate_group_isolation(
    splits: dict[str, Sequence[Sample]],
    dataset_root: str | Path,
    path_to_group: dict[str, str],
) -> None:
    root = Path(dataset_root).resolve()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, split_samples in splits.items():
        for sample in split_samples:
            relative = sample.path.resolve().relative_to(root).as_posix()
            group_splits[path_to_group[relative]].add(split_name)
    crossing = {
        group_id: sorted(names) for group_id, names in group_splits.items() if len(names) > 1
    }
    if crossing:
        raise ValueError(f"Groups cross split boundaries: {crossing}")


def grouping_summary(path_to_group: dict[str, str]) -> dict[str, Any]:
    members: dict[str, list[str]] = defaultdict(list)
    for relative_path, group_id in path_to_group.items():
        members[group_id].append(relative_path)
    multi = {group_id: sorted(paths) for group_id, paths in members.items() if len(paths) > 1}
    return {
        "total_groups": len(members),
        "multi_image_groups": len(multi),
        "grouped_images": sum(len(paths) for paths in multi.values()),
        "groups": dict(sorted(multi.items())),
    }
