"""Independent RESISC45 calibration, stability, and OOD evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terraclass.config import load_config
from terraclass.data import ImagePathDataset, file_sha256, load_manifest
from terraclass.inference import TerraClassPredictor, load_serving_config
from terraclass.model_quality import (
    calibration_metrics,
    fit_temperature,
    render_reliability_figure,
)

MANIFEST_FIELDS = (
    "role",
    "source_split",
    "relative_path",
    "source_class",
    "target_class",
    "label",
    "sha256",
)


@dataclass(frozen=True)
class ExternalManifestRow:
    role: str
    source_split: str
    relative_path: str
    source_class: str
    target_class: str | None
    label: int | None
    sha256: str


class ExternalImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        rows: list[ExternalManifestRow],
        dataset_root: Path,
        transform: Any,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.dataset_root / row.relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, -1 if row.label is None else row.label


def load_external_calibration_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for field in (
        "serving_config_path",
        "uc_merced_dataset_root",
        "manifest_path",
    ):
        value = config.get(field)
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            errors.append(f"{field} must be project-relative")
    external = config.get("external_dataset", {})
    if external.get("name") != "NWPU-RESISC45":
        errors.append("external dataset must be NWPU-RESISC45")
    archive = external.get("archive", {})
    if (
        archive.get("size_bytes") != 427_389_445
        or archive.get("sha256")
        != "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
    ):
        errors.append("external archive identity is not pinned")
    redistribution = external.get("redistribution", {})
    if (
        redistribution.get("stated_license") != "CC-BY-NC-4.0"
        or redistribution.get("commercial_use_permitted") is not False
    ):
        errors.append("external redistribution license boundary is invalid")
    expected_targets = (
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
    )
    mapping = config.get("class_mapping", {})
    if tuple(mapping) != expected_targets:
        errors.append("class mapping must preserve the serving class order")
    flattened = [source for sources in mapping.values() for source in sources]
    if len(flattened) != len(set(flattened)):
        errors.append("one external source class cannot map to multiple target classes")
    sampling = config.get("sampling", {})
    if (
        sampling.get("calibration_source_split") != "validation"
        or sampling.get("evaluation_source_split") != "test"
        or sampling.get("ood_source_split") != "test"
        or sampling.get("aligned_samples_per_target_class") != 100
    ):
        errors.append("external sampling roles are invalid")
    calibration = config.get("calibration", {})
    lower, upper = calibration.get("temperature_bounds", (0, 0))
    if (
        not 0 < lower < upper
        or calibration.get("bootstrap_replicates", 0) < 100
        or calibration.get("cross_validation_folds", 0) < 3
    ):
        errors.append("external calibration stability configuration is invalid")
    gates = config.get("promotion_gates", {})
    claims = config.get("claim_boundary", {})
    if gates.get("automatic_production_promotion") is not False:
        errors.append("automatic production promotion must remain disabled")
    if claims != {
        "external_domain_is_production_representative": False,
        "external_mapping_is_exact": False,
        "license_approved_for_commercial_production": False,
        "temperature_scaling_is_ood_detector": False,
        "iit_submission_changed": False,
    }:
        errors.append("external calibration claim boundary is invalid")
    if errors:
        raise ValueError("Invalid external calibration configuration: " + "; ".join(errors))
    return config


def source_class_from_filename(filename: str) -> str:
    path = Path(filename)
    stem, separator, index = path.stem.rpartition("_")
    if (
        path.name != filename
        or path.suffix.lower() != ".jpg"
        or separator != "_"
        or not stem
        or not index.isdigit()
    ):
        raise ValueError(f"Invalid RESISC45 filename: {filename}")
    return stem


def _ranked_take(filenames: list[str], count: int, seed: int) -> list[str]:
    if count < 0 or count > len(filenames):
        raise ValueError(f"Cannot select {count} items from {len(filenames)} candidates")
    return sorted(
        filenames,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )[:count]


def select_aligned_filenames(
    filenames: list[str],
    class_mapping: dict[str, list[str]],
    *,
    per_target_class: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    """Select a deterministic, balanced external sample for the five serving labels."""
    by_source: dict[str, list[str]] = {}
    for filename in filenames:
        by_source.setdefault(source_class_from_filename(filename), []).append(filename)
    selected: list[tuple[str, str, int]] = []
    for label, (target, sources) in enumerate(class_mapping.items()):
        if per_target_class % len(sources):
            raise ValueError(f"{target} sample count is not divisible by its source-class count")
        per_source = per_target_class // len(sources)
        for source_index, source in enumerate(sources):
            candidates = by_source.get(source, [])
            chosen = _ranked_take(candidates, per_source, seed + label * 100 + source_index)
            selected.extend((filename, target, label) for filename in chosen)
    if len(selected) != per_target_class * len(class_mapping):
        raise RuntimeError("Aligned external selection has an unexpected size")
    return selected


def _read_split(path: Path, expected_sha256: str, expected_rows: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"RESISC45 split file does not exist: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"RESISC45 split hash differs from the configured value: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or len(rows) != len(set(rows)):
        raise ValueError(f"RESISC45 split row contract failed: {path}")
    return rows


def build_external_manifest(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> list[ExternalManifestRow]:
    root = project_root.resolve()
    external = config["external_dataset"]
    dataset_root = (root / external["root"]).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"RESISC45 dataset root does not exist: {dataset_root}")
    split_rows = {
        name: _read_split(
            root / split["path"],
            split["sha256"],
            int(split["rows"]),
        )
        for name, split in external["splits"].items()
    }
    mapping = config["class_mapping"]
    sampling = config["sampling"]
    seed = int(sampling["seed"])
    per_target = int(sampling["aligned_samples_per_target_class"])
    selected: list[ExternalManifestRow] = []
    role_specs = (
        ("calibration", sampling["calibration_source_split"], seed),
        ("external_test", sampling["evaluation_source_split"], seed + 10_000),
    )
    for role, source_split, role_seed in role_specs:
        aligned = select_aligned_filenames(
            split_rows[source_split],
            mapping,
            per_target_class=per_target,
            seed=role_seed,
        )
        for filename, target, label in aligned:
            source_class = source_class_from_filename(filename)
            image_path = dataset_root / source_class / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"RESISC45 image does not exist: {image_path}")
            selected.append(
                ExternalManifestRow(
                    role=role,
                    source_split=source_split,
                    relative_path=image_path.relative_to(dataset_root).as_posix(),
                    source_class=source_class,
                    target_class=target,
                    label=label,
                    sha256=file_sha256(image_path),
                )
            )

    mapped_sources = {source for sources in mapping.values() for source in sources}
    ood_split = sampling["ood_source_split"]
    ood_filenames = sorted(
        filename
        for filename in split_rows[ood_split]
        if source_class_from_filename(filename) not in mapped_sources
    )
    for filename in ood_filenames:
        source_class = source_class_from_filename(filename)
        image_path = dataset_root / source_class / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"RESISC45 OOD image does not exist: {image_path}")
        selected.append(
            ExternalManifestRow(
                role="ood_test",
                source_split=ood_split,
                relative_path=image_path.relative_to(dataset_root).as_posix(),
                source_class=source_class,
                target_class=None,
                label=None,
                sha256=file_sha256(image_path),
            )
        )
    paths_by_role: dict[str, set[str]] = {}
    for row in selected:
        paths_by_role.setdefault(row.role, set()).add(row.relative_path)
    roles = list(paths_by_role)
    for index, first in enumerate(roles):
        for second in roles[index + 1 :]:
            overlap = paths_by_role[first] & paths_by_role[second]
            if overlap:
                raise ValueError(f"External manifest roles overlap: {first}/{second}")

    manifest_path = root / config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "role": row.role,
                    "source_split": row.source_split,
                    "relative_path": row.relative_path,
                    "source_class": row.source_class,
                    "target_class": row.target_class or "",
                    "label": "" if row.label is None else row.label,
                    "sha256": row.sha256,
                }
            )
    return selected


def load_external_manifest(
    path: Path,
    dataset_root: Path,
    *,
    verify_hashes: bool,
) -> list[ExternalManifestRow]:
    rows: list[ExternalManifestRow] = []
    root = dataset_root.resolve()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("External calibration manifest columns are invalid")
        for raw in reader:
            relative = Path(raw["relative_path"])
            resolved = (root / relative).resolve()
            if relative.is_absolute() or (resolved != root and root not in resolved.parents):
                raise ValueError("External manifest path escapes the dataset root")
            if not resolved.is_file():
                raise FileNotFoundError(f"External manifest image does not exist: {resolved}")
            if verify_hashes and file_sha256(resolved) != raw["sha256"]:
                raise ValueError(f"External image hash mismatch: {relative}")
            rows.append(
                ExternalManifestRow(
                    role=raw["role"],
                    source_split=raw["source_split"],
                    relative_path=relative.as_posix(),
                    source_class=raw["source_class"],
                    target_class=raw["target_class"] or None,
                    label=int(raw["label"]) if raw["label"] else None,
                    sha256=raw["sha256"],
                )
            )
    return rows


def collect_external_logits(
    model: nn.Module,
    dataset: ExternalImageDataset,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def bootstrap_temperature_interval(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bounds: tuple[float, float],
    iterations: int,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if replicates < 100 or not 0 < confidence_level < 1:
        raise ValueError("bootstrap requires at least 100 replicates and confidence in (0, 1)")
    generator = np.random.default_rng(seed)
    label_values = labels.numpy()
    class_indices = [np.flatnonzero(label_values == label) for label in sorted(set(label_values))]
    temperatures: list[float] = []
    bound_hits = 0
    for _ in range(replicates):
        sampled = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        result = fit_temperature(
            logits[sampled],
            labels[sampled],
            bounds=bounds,
            iterations=iterations,
        )
        temperatures.append(float(result["temperature"]))
        bound_hits += int(bool(result["at_optimization_bound"]))
    values = np.asarray(temperatures)
    alpha = (1 - confidence_level) / 2
    lower, median, upper = np.quantile(values, (alpha, 0.5, 1 - alpha))
    relative_width = math.inf if median == 0 else (upper - lower) / median
    return {
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "temperature": {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
            "relative_interval_width": float(relative_width),
            "bound_hits": bound_hits,
        },
    }


def cross_validated_temperature_stability(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bounds: tuple[float, float],
    iterations: int,
    folds: int,
    bin_count: int,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    label_values = labels.numpy()
    rows: list[dict[str, Any]] = []
    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), label_values),
        start=1,
    ):
        fit = fit_temperature(
            logits[fit_indices],
            labels[fit_indices],
            bounds=bounds,
            iterations=iterations,
        )
        temperature = float(fit["temperature"])
        before = calibration_metrics(
            logits[holdout_indices],
            labels[holdout_indices],
            bin_count=bin_count,
        )
        after = calibration_metrics(
            logits[holdout_indices] / temperature,
            labels[holdout_indices],
            bin_count=bin_count,
        )
        rows.append(
            {
                "fold": fold,
                "fit_samples": len(fit_indices),
                "holdout_samples": len(holdout_indices),
                "temperature": temperature,
                "at_optimization_bound": fit["at_optimization_bound"],
                "holdout_nll_before": before["negative_log_likelihood"],
                "holdout_nll_after": after["negative_log_likelihood"],
                "holdout_brier_before": before["multiclass_brier_score"],
                "holdout_brier_after": after["multiclass_brier_score"],
                "holdout_ece_before": before["expected_calibration_error"],
                "holdout_ece_after": after["expected_calibration_error"],
            }
        )
    temperatures = np.asarray([row["temperature"] for row in rows])
    coefficient_of_variation = float(temperatures.std(ddof=1) / temperatures.mean())
    return {
        "folds": folds,
        "seed": seed,
        "temperature_mean": float(temperatures.mean()),
        "temperature_standard_deviation": float(temperatures.std(ddof=1)),
        "temperature_coefficient_of_variation": coefficient_of_variation,
        "all_fits_interior": not any(row["at_optimization_bound"] for row in rows),
        "all_holdout_nll_improved": all(
            row["holdout_nll_after"] < row["holdout_nll_before"] for row in rows
        ),
        "results": rows,
    }


def ood_detection_metrics(
    aligned_logits: torch.Tensor,
    ood_logits: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, Any]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    aligned_probabilities = torch.softmax(aligned_logits.double() / temperature, dim=1)
    ood_probabilities = torch.softmax(ood_logits.double() / temperature, dim=1)
    class_count = aligned_logits.shape[1]
    aligned_confidence = aligned_probabilities.max(dim=1).values.numpy()
    ood_confidence = ood_probabilities.max(dim=1).values.numpy()
    aligned_entropy = (
        -(aligned_probabilities * aligned_probabilities.clamp_min(1e-15).log()).sum(dim=1)
        / math.log(class_count)
    ).numpy()
    ood_entropy = (
        -(ood_probabilities * ood_probabilities.clamp_min(1e-15).log()).sum(dim=1)
        / math.log(class_count)
    ).numpy()
    targets = np.concatenate((np.ones(len(aligned_logits)), np.zeros(len(ood_logits))))

    def summarize(aligned_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
        scores = np.concatenate((aligned_scores, ood_scores))
        threshold = float(np.quantile(aligned_scores, 0.05))
        return {
            "auroc": float(roc_auc_score(targets, scores)),
            "average_precision": float(average_precision_score(targets, scores)),
            "fpr_at_95_tpr": float(np.mean(ood_scores >= threshold)),
            "threshold_at_95_tpr": threshold,
            "aligned_mean": float(aligned_scores.mean()),
            "ood_mean": float(ood_scores.mean()),
        }

    return {
        "aligned_samples": len(aligned_logits),
        "ood_samples": len(ood_logits),
        "temperature": temperature,
        "maximum_softmax_probability": summarize(aligned_confidence, ood_confidence),
        "negative_normalized_entropy": summarize(-aligned_entropy, -ood_entropy),
        "claim_boundary": "Post-hoc temperature scaling is not treated as an OOD detector.",
    }


def evaluate_promotion_gates(
    *,
    config: dict[str, Any],
    fit: dict[str, Any],
    bootstrap: dict[str, Any],
    cross_validation: dict[str, Any],
    calibration_before: dict[str, Any],
    external_before: dict[str, Any],
    external_after: dict[str, Any],
    in_domain_before: dict[str, Any],
    in_domain_after: dict[str, Any],
) -> dict[str, Any]:
    gates = config["promotion_gates"]
    calibration_errors = calibration_before["sample_count"] * (1 - calibration_before["accuracy"])
    in_domain_nll_before = in_domain_before["negative_log_likelihood"]
    in_domain_nll_after = in_domain_after["negative_log_likelihood"]
    in_domain_relative_degradation = (
        in_domain_nll_after - in_domain_nll_before
    ) / in_domain_nll_before
    checks = {
        "minimum_calibration_samples": (
            calibration_before["sample_count"] >= gates["minimum_calibration_samples"]
        ),
        "minimum_calibration_errors": (calibration_errors >= gates["minimum_calibration_errors"]),
        "interior_temperature": not fit["at_optimization_bound"],
        "bootstrap_interval_width": (
            bootstrap["temperature"]["relative_interval_width"]
            <= gates["maximum_bootstrap_relative_interval_width"]
        ),
        "bootstrap_no_bound_hits": bootstrap["temperature"]["bound_hits"] == 0,
        "cross_validation_temperature_cv": (
            cross_validation["temperature_coefficient_of_variation"]
            <= gates["maximum_cross_validation_temperature_cv"]
        ),
        "cross_validation_all_fits_interior": cross_validation["all_fits_interior"],
        "cross_validation_all_holdout_nll_improved": (cross_validation["all_holdout_nll_improved"]),
        "external_test_nll_improved": (
            external_after["negative_log_likelihood"] < external_before["negative_log_likelihood"]
        ),
        "external_test_brier_improved": (
            external_after["multiclass_brier_score"] < external_before["multiclass_brier_score"]
        ),
        "external_test_ece_improved": (
            external_after["expected_calibration_error"]
            < external_before["expected_calibration_error"]
        ),
        "external_test_classification_unchanged": (
            external_after["accuracy"] == external_before["accuracy"]
            and external_after["macro_f1"] == external_before["macro_f1"]
        ),
        "in_domain_classification_unchanged": (
            in_domain_after["accuracy"] == in_domain_before["accuracy"]
            and in_domain_after["macro_f1"] == in_domain_before["macro_f1"]
        ),
        "in_domain_nll_degradation_within_limit": (
            in_domain_relative_degradation <= gates["maximum_in_domain_nll_relative_degradation"]
        ),
    }
    statistical_gates_passed = all(checks.values())
    claims = config["claim_boundary"]
    policy_blockers = {
        "external_domain_not_proven_production_representative": not claims[
            "external_domain_is_production_representative"
        ],
        "external_class_mapping_contains_proxies": not claims["external_mapping_is_exact"],
        "license_not_approved_for_commercial_production": not claims[
            "license_approved_for_commercial_production"
        ],
        "automatic_production_promotion_disabled": not gates["automatic_production_promotion"],
    }
    return {
        "calibration_errors": int(round(calibration_errors)),
        "in_domain_nll_relative_degradation": in_domain_relative_degradation,
        "checks": checks,
        "statistical_gates_passed": statistical_gates_passed,
        "policy_blockers": policy_blockers,
        "production_promotion_approved": (
            statistical_gates_passed and not any(policy_blockers.values())
        ),
    }


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_external_calibration(
    *,
    project_root: Path,
    config_path: Path,
    output_path: Path,
    figure_path: Path,
    device: str,
    report_date: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_external_calibration_config(config_path)
    manifest_path = root / config["manifest_path"]
    build_external_manifest(project_root=root, config=config)
    external_root = root / config["external_dataset"]["root"]
    rows = load_external_manifest(manifest_path, external_root, verify_hashes=True)
    by_role = {
        role: [row for row in rows if row.role == role]
        for role in ("calibration", "external_test", "ood_test")
    }
    serving = load_serving_config(root / config["serving_config_path"])
    predictor = TerraClassPredictor.load(serving, root, device=device)

    calibration_logits, calibration_labels = collect_external_logits(
        predictor.model,
        ExternalImageDataset(by_role["calibration"], external_root, predictor.transform),
        predictor.device,
    )
    external_test_logits, external_test_labels = collect_external_logits(
        predictor.model,
        ExternalImageDataset(by_role["external_test"], external_root, predictor.transform),
        predictor.device,
    )
    ood_logits, _ = collect_external_logits(
        predictor.model,
        ExternalImageDataset(by_role["ood_test"], external_root, predictor.transform),
        predictor.device,
    )

    baseline = load_config(root / serving.baseline_config_path)
    uc_splits, _ = load_manifest(
        root / serving.training_manifest_path,
        root / config["uc_merced_dataset_root"],
        baseline,
        verify_hashes=True,
    )
    in_domain_dataset = ImagePathDataset(
        uc_splits["test"],
        predictor.transform,
    )
    in_domain_loader = DataLoader(in_domain_dataset, batch_size=32, shuffle=False, num_workers=0)
    in_domain_logits: list[torch.Tensor] = []
    in_domain_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in in_domain_loader:
            in_domain_logits.append(predictor.model(images.to(predictor.device)).cpu())
            in_domain_labels.append(labels)
    uc_test_logits = torch.cat(in_domain_logits)
    uc_test_labels = torch.cat(in_domain_labels)

    calibration_config = config["calibration"]
    bounds = tuple(float(value) for value in calibration_config["temperature_bounds"])
    iterations = int(calibration_config["optimization_iterations"])
    bin_count = int(calibration_config["bin_count"])
    seed = int(config["sampling"]["seed"])
    fit = fit_temperature(
        calibration_logits,
        calibration_labels,
        bounds=bounds,
        iterations=iterations,
    )
    temperature = float(fit["temperature"])
    calibration_before = calibration_metrics(
        calibration_logits,
        calibration_labels,
        bin_count=bin_count,
    )
    calibration_after = calibration_metrics(
        calibration_logits / temperature,
        calibration_labels,
        bin_count=bin_count,
    )
    external_before = calibration_metrics(
        external_test_logits,
        external_test_labels,
        bin_count=bin_count,
    )
    external_after = calibration_metrics(
        external_test_logits / temperature,
        external_test_labels,
        bin_count=bin_count,
    )
    in_domain_before = calibration_metrics(
        uc_test_logits,
        uc_test_labels,
        bin_count=bin_count,
    )
    in_domain_after = calibration_metrics(
        uc_test_logits / temperature,
        uc_test_labels,
        bin_count=bin_count,
    )
    bootstrap = bootstrap_temperature_interval(
        calibration_logits,
        calibration_labels,
        bounds=bounds,
        iterations=iterations,
        replicates=int(calibration_config["bootstrap_replicates"]),
        confidence_level=float(calibration_config["bootstrap_confidence_level"]),
        seed=seed,
    )
    cross_validation = cross_validated_temperature_stability(
        calibration_logits,
        calibration_labels,
        bounds=bounds,
        iterations=iterations,
        folds=int(calibration_config["cross_validation_folds"]),
        bin_count=bin_count,
        seed=seed,
    )
    promotion = evaluate_promotion_gates(
        config=config,
        fit=fit,
        bootstrap=bootstrap,
        cross_validation=cross_validation,
        calibration_before=calibration_before,
        external_before=external_before,
        external_after=external_after,
        in_domain_before=in_domain_before,
        in_domain_after=in_domain_after,
    )
    render_reliability_figure(
        external_before,
        external_after,
        figure_path,
        title=(
            "RESISC45 external-domain reliability "
            f"({len(external_test_labels)}-image untouched test)"
        ),
        before_title="Original softmax",
        after_title=f"Temperature candidate (T={temperature:.3f})",
    )
    ood_before = ood_detection_metrics(
        external_test_logits,
        ood_logits,
        temperature=1.0,
    )
    ood_after = ood_detection_metrics(
        external_test_logits,
        ood_logits,
        temperature=temperature,
    )
    report = {
        "schema_version": 1,
        "evaluated_on": report_date,
        "evaluation_id": config["evaluation_id"],
        "environment": _environment(),
        "model": {
            "model_id": serving.model_id,
            "model_version": serving.model_version,
            "architecture": serving.architecture,
            "class_names": list(serving.class_names),
            "serving_artifact_sha256": serving.serving_artifact.sha256,
        },
        "external_dataset": {
            "name": config["external_dataset"]["name"],
            "archive_sha256": config["external_dataset"]["archive"]["sha256"],
            "archive_size_bytes": config["external_dataset"]["archive"]["size_bytes"],
            "license": config["external_dataset"]["redistribution"]["stated_license"],
            "commercial_use_permitted": False,
            "citation": config["external_dataset"]["citation"],
            "class_mapping": config["class_mapping"],
            "mapping_basis": config["mapping_basis"],
        },
        "provenance": {
            "config_path": config_path.resolve().relative_to(root).as_posix(),
            "config_sha256": file_sha256(config_path),
            "manifest_path": config["manifest_path"],
            "manifest_sha256": file_sha256(manifest_path),
            "role_counts": {role: len(role_rows) for role, role_rows in by_role.items()},
            "role_source_splits": {
                role: sorted({row.source_split for row in role_rows})
                for role, role_rows in by_role.items()
            },
            "roles_disjoint": True,
        },
        "temperature_scaling": {
            "fit_split": "RESISC45 official validation",
            "fit": fit,
            "bootstrap": bootstrap,
            "cross_validation": cross_validation,
            "calibration": {
                "before": calibration_before,
                "after": calibration_after,
            },
            "untouched_external_test": {
                "before": external_before,
                "after": external_after,
            },
            "uc_merced_group_aware_test_sensitivity": {
                "before": in_domain_before,
                "after": in_domain_after,
            },
        },
        "ood_detection": {
            "definition": "All 39 unmapped RESISC45 classes from the official test split",
            "before_temperature": ood_before,
            "after_temperature": ood_after,
        },
        "promotion": promotion,
        "figure": {
            "path": figure_path.resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(figure_path),
        },
        "claim_boundary": {
            **config["claim_boundary"],
            "production_model_changed": False,
            "calibration_candidate_only": True,
            "reason": (
                "Statistical evidence is evaluated separately from production-domain, semantic "
                "mapping, and license approval."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/external_calibration_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/external_calibration_evaluation_2026-07-18.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/external_calibration_resisc45_2026-07-18.png"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--report-date", default="2026-07-18")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    report = run_external_calibration(
        project_root=root,
        config_path=rooted(args.config),
        output_path=rooted(args.output),
        figure_path=rooted(args.figure),
        device=args.device,
        report_date=args.report_date,
    )
    test = report["temperature_scaling"]["untouched_external_test"]
    print(
        json.dumps(
            {
                "temperature": report["temperature_scaling"]["fit"]["temperature"],
                "external_test_before": test["before"],
                "external_test_after": test["after"],
                "statistical_gates_passed": report["promotion"]["statistical_gates_passed"],
                "production_promotion_approved": report["promotion"][
                    "production_promotion_approved"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
