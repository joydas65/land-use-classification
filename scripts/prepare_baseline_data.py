"""Audit the downloaded dataset and create the versioned five-class split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terraclass.config import load_config
from terraclass.data import class_counts, discover_samples, stratified_split, write_manifest
from terraclass.dataset_audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/UCMerced_LandUse/Images"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_5class.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/DATASET_AUDIT.json"))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/manifests/baseline_5class_seed42.csv"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = audit_dataset(args.dataset_root, config)
    samples = discover_samples(args.dataset_root, config)
    splits = stratified_split(samples, config)
    write_manifest(
        args.manifest_output,
        splits,
        args.dataset_root,
        include_hashes=True,
    )
    split_by_path = {
        sample.path.resolve().relative_to(args.dataset_root.resolve()).as_posix(): split_name
        for split_name, split_samples in splits.items()
        for sample in split_samples
    }
    selected_candidates = [
        item
        for item in report["perceptual_hash"]["candidate_examples"]
        if item["left"] in split_by_path and item["right"] in split_by_path
    ]
    for item in selected_candidates:
        item["left_split"] = split_by_path[item["left"]]
        item["right_split"] = split_by_path[item["right"]]
    cross_split_candidates = [
        item for item in selected_candidates if item["left_split"] != item["right_split"]
    ]
    report["baseline_split"] = {
        "seed": config.seed,
        "manifest": args.manifest_output.as_posix(),
        "counts": {name: len(values) for name, values in splits.items()},
        "class_counts": {name: class_counts(values) for name, values in splits.items()},
        "perceptual_candidate_pair_count": len(selected_candidates),
        "perceptual_candidate_cross_split_count": len(cross_split_candidates),
        "perceptual_candidate_cross_split_examples": cross_split_candidates,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
