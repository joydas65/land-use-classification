"""Create the reviewed, group-aware five-class split manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from terraclass.config import load_config
from terraclass.data import discover_samples, file_sha256, write_manifest
from terraclass.grouping import (
    group_aware_stratified_split,
    grouping_summary,
    load_verified_groups,
    validate_group_isolation,
)


def historical_group_crossings(
    historical_manifest: Path, reviewed_groups: dict[str, str]
) -> dict[str, list[str]]:
    """Report reviewed groups that cross the supplied notebook's split boundaries."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    with historical_manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group_id = reviewed_groups.get(row["relative_path"])
            if group_id:
                group_splits[group_id].add(row["split"])
    return {
        group_id: sorted(split_names)
        for group_id, split_names in sorted(group_splits.items())
        if len(split_names) > 1
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_5class.json"))
    parser.add_argument(
        "--groups", type=Path, default=Path("data/grouping/verified_related_scenes.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/five_class_group_aware_seed42.csv"),
    )
    parser.add_argument("--audit-output", type=Path, default=Path("data/GROUP_AWARE_AUDIT.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    reviewed_groups = load_verified_groups(args.groups)
    samples = discover_samples(args.dataset_root, config)
    splits, path_to_group = group_aware_stratified_split(
        samples, config, args.dataset_root, reviewed_groups
    )
    validate_group_isolation(splits, args.dataset_root, path_to_group)
    write_manifest(
        args.output,
        splits,
        args.dataset_root,
        include_hashes=True,
        group_by_relative_path=path_to_group,
    )
    audit = {
        "schema_version": 1,
        "seed": config.seed,
        "selected_classes": list(config.dataset.selected_classes),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "manifest_path": args.output.as_posix(),
        "manifest_sha256": file_sha256(args.output),
        "reviewed_groups_path": args.groups.as_posix(),
        "reviewed_groups_sha256": file_sha256(args.groups),
        "historical_manifest_sha256": config.split.manifest_sha256,
        "historical_reviewed_group_crossings": historical_group_crossings(
            Path(config.split.manifest_path), reviewed_groups
        ),
        "group_aware_crossings": {},
        "grouping": grouping_summary(path_to_group),
    }
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
