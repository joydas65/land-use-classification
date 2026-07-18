"""Deterministic corruption robustness and validation-selected TTA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terraclass.config import load_config
from terraclass.data import Sample, file_sha256, load_manifest
from terraclass.inference import TerraClassPredictor, load_serving_config
from terraclass.model_quality import calibration_metrics

CORRUPTION_NAMES = (
    "brightness_reduction",
    "contrast_reduction",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
)
TTA_TRANSFORMS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
)
SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "expected_calibration_error",
    "mean_confidence",
)


class CorruptedImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        corruption: str | None,
        parameter: float | int | None,
        severity: int,
        seed: int,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.corruption = corruption
        self.parameter = parameter
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        if self.corruption is not None:
            identity = f"{sample.class_name}/{sample.path.name}"
            image = apply_corruption(
                image,
                self.corruption,
                self.parameter,
                seed=deterministic_sample_seed(self.seed, identity, self.severity),
            )
        return self.transform(image), sample.label


def load_robustness_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not str(config.get("evaluation_id", "")).strip():
        errors.append("evaluation_id must not be blank")
    for field in ("serving_config_path", "dataset_root"):
        value = config.get(field)
        path_value = Path(value) if isinstance(value, str) else Path("/")
        if not isinstance(value, str) or path_value.is_absolute() or ".." in path_value.parts:
            errors.append(f"{field} must be project-relative")
    protocol = config.get("protocol", {})
    corruptions = protocol.get("corruptions", {})
    if (
        protocol.get("selection_split") != "validation"
        or protocol.get("evaluation_split") != "test"
        or int(protocol.get("batch_size", 0)) <= 0
        or int(protocol.get("bin_count", 0)) < 2
        or tuple(corruptions) != CORRUPTION_NAMES
        or any(len(values) != 3 for values in corruptions.values())
    ):
        errors.append("robustness protocol is invalid")
    candidate = config.get("candidate", {})
    if (
        candidate.get("name") != "dihedral_4_logit_mean"
        or tuple(candidate.get("transforms", ())) != TTA_TRANSFORMS
        or candidate.get("aggregation") != "mean_logits"
        or candidate.get("selection_metric") != "mean_corruption_macro_f1"
        or float(candidate.get("minimum_validation_improvement", -1)) < 0
        or candidate.get("require_clean_classification_unchanged") is not True
        or candidate.get("require_worst_corruption_macro_f1_not_worse") is not True
    ):
        errors.append("TTA candidate contract is invalid")
    gates = config.get("promotion_gates", {})
    if gates != {
        "require_candidate_selected_on_validation": True,
        "require_test_clean_classification_unchanged": True,
        "require_test_mean_corruption_macro_f1_improvement": True,
        "require_test_worst_corruption_macro_f1_not_worse": True,
        "automatic_production_promotion": False,
    }:
        errors.append("robustness promotion gates are invalid")
    if config.get("claim_boundary") != {
        "synthetic_corruptions_are_production_representative": False,
        "test_split_used_for_candidate_selection": False,
        "adversarial_robustness_evaluated": False,
        "resisc45_used": False,
        "iit_submission_changed": False,
    }:
        errors.append("robustness claim boundary is invalid")
    if errors:
        raise ValueError("Invalid robustness configuration: " + "; ".join(errors))
    return config


def deterministic_sample_seed(
    base_seed: int,
    sample_identity: str,
    severity: int,
) -> int:
    if severity < 0:
        raise ValueError("severity must be non-negative")
    digest = hashlib.sha256(f"{base_seed}:{severity}:{sample_identity}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def apply_corruption(
    image: Image.Image,
    name: str,
    parameter: float | int | None,
    *,
    seed: int,
) -> Image.Image:
    """Apply one deterministic, label-preserving synthetic corruption."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if name not in CORRUPTION_NAMES or parameter is None:
        raise ValueError(f"Unsupported corruption: {name}")
    value = float(parameter)
    if name == "brightness_reduction":
        if not 0 < value <= 1:
            raise ValueError("brightness factor must be in (0, 1]")
        return ImageEnhance.Brightness(image).enhance(value)
    if name == "contrast_reduction":
        if not 0 < value <= 1:
            raise ValueError("contrast factor must be in (0, 1]")
        return ImageEnhance.Contrast(image).enhance(value)
    if name == "gaussian_blur":
        if value <= 0:
            raise ValueError("blur radius must be positive")
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if name == "gaussian_noise":
        if not 0 < value <= 0.25:
            raise ValueError("noise standard deviation must be in (0, 0.25]")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        generator = np.random.default_rng(seed)
        noisy = np.clip(
            pixels + generator.normal(0.0, value, size=pixels.shape),
            0.0,
            1.0,
        )
        return Image.fromarray(np.rint(noisy * 255).astype(np.uint8))
    quality = int(parameter)
    if quality != parameter or not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be an integer in [1, 95]")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def tta_variant(images: torch.Tensor, transform_name: str) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("TTA expects a [batch, channels, height, width] tensor")
    if transform_name == "identity":
        return images
    if transform_name == "horizontal_flip":
        return torch.flip(images, dims=(-1,))
    if transform_name == "vertical_flip":
        return torch.flip(images, dims=(-2,))
    if transform_name == "rotate_180":
        return torch.flip(images, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA transform: {transform_name}")


def corruption_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = [
        {
            "id": "clean",
            "corruption": None,
            "severity": 0,
            "parameter": None,
        }
    ]
    for corruption, parameters in config["protocol"]["corruptions"].items():
        for severity, parameter in enumerate(parameters, start=1):
            scenarios.append(
                {
                    "id": f"{corruption}:severity_{severity}",
                    "corruption": corruption,
                    "severity": severity,
                    "parameter": parameter,
                }
            )
    return scenarios


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_policy_logits(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    device: torch.device,
    *,
    batch_size: int,
    include_candidate: bool,
    transforms: Sequence[str] = TTA_TRANSFORMS,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    baseline_logits: list[torch.Tensor] = []
    candidate_logits: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            _synchronize(device)
            started = time.perf_counter()
            identity_logits = model(images)
            _synchronize(device)
            identity_seconds = time.perf_counter() - started
            baseline_seconds += identity_seconds
            baseline_logits.append(identity_logits.cpu())
            labels_rows.append(labels)
            if include_candidate:
                logits_rows = [identity_logits]
                _synchronize(device)
                candidate_started = time.perf_counter()
                for transform_name in transforms[1:]:
                    logits_rows.append(model(tta_variant(images, transform_name)))
                _synchronize(device)
                candidate_seconds += identity_seconds + (time.perf_counter() - candidate_started)
                candidate_logits.append(torch.stack(logits_rows).mean(dim=0).cpu())
    labels = torch.cat(labels_rows)
    return {
        "labels": labels,
        "baseline_logits": torch.cat(baseline_logits),
        "candidate_logits": (torch.cat(candidate_logits) if include_candidate else None),
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds if include_candidate else None,
    }


def _average_metrics(rows: Sequence[dict[str, Any]], policy: str) -> dict[str, float]:
    policy_rows = [row[policy] for row in rows if row.get(policy) is not None]
    if not policy_rows:
        return {}
    return {
        metric: float(np.mean([row[metric] for row in policy_rows])) for metric in SUMMARY_METRICS
    }


def _policy_summary(
    scenarios: Sequence[dict[str, Any]],
    policy: str,
) -> dict[str, Any] | None:
    if not any(row.get(policy) is not None for row in scenarios):
        return None
    clean = next(row for row in scenarios if row["id"] == "clean")
    corrupted = [row for row in scenarios if row["severity"] > 0]
    worst = min(
        (row for row in corrupted if row.get(policy) is not None),
        key=lambda row: (row[policy]["macro_f1"], row[policy]["accuracy"], row["id"]),
    )
    by_severity = [
        {
            "severity": severity,
            **_average_metrics(
                [row for row in corrupted if row["severity"] == severity],
                policy,
            ),
        }
        for severity in (1, 2, 3)
    ]
    by_corruption = {
        corruption: _average_metrics(
            [row for row in corrupted if row["corruption"] == corruption],
            policy,
        )
        for corruption in CORRUPTION_NAMES
    }
    total_samples = sum(int(row[policy]["sample_count"]) for row in scenarios)
    total_seconds = sum(float(row[f"{policy}_seconds"]) for row in scenarios)
    return {
        "clean": clean[policy],
        "corruption_average": _average_metrics(corrupted, policy),
        "by_severity": by_severity,
        "by_corruption": by_corruption,
        "worst_condition": {
            "id": worst["id"],
            "corruption": worst["corruption"],
            "severity": worst["severity"],
            "parameter": worst["parameter"],
            "accuracy": worst[policy]["accuracy"],
            "macro_f1": worst[policy]["macro_f1"],
        },
        "inference": {
            "evaluated_samples": total_samples,
            "elapsed_seconds": total_seconds,
            "mean_ms_per_image": total_seconds / total_samples * 1000,
        },
    }


def summarize_scenarios(scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline = _policy_summary(scenarios, "baseline")
    candidate = _policy_summary(scenarios, "candidate")
    if baseline is None:
        raise ValueError("baseline scenarios are required")
    summary = {
        "scenario_count": len(scenarios),
        "corrupted_scenario_count": sum(row["severity"] > 0 for row in scenarios),
        "baseline": baseline,
        "candidate": candidate,
    }
    if candidate is not None:
        summary["candidate_vs_baseline"] = {
            "clean_accuracy_delta": (
                candidate["clean"]["accuracy"] - baseline["clean"]["accuracy"]
            ),
            "clean_macro_f1_delta": (
                candidate["clean"]["macro_f1"] - baseline["clean"]["macro_f1"]
            ),
            "mean_corruption_accuracy_delta": (
                candidate["corruption_average"]["accuracy"]
                - baseline["corruption_average"]["accuracy"]
            ),
            "mean_corruption_macro_f1_delta": (
                candidate["corruption_average"]["macro_f1"]
                - baseline["corruption_average"]["macro_f1"]
            ),
            "worst_corruption_macro_f1_delta": (
                candidate["worst_condition"]["macro_f1"] - baseline["worst_condition"]["macro_f1"]
            ),
            "latency_multiplier": (
                candidate["inference"]["mean_ms_per_image"]
                / baseline["inference"]["mean_ms_per_image"]
            ),
        }
    return summary


def evaluate_candidate_selection(
    validation_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate = validation_summary.get("candidate")
    if candidate is None:
        raise ValueError("validation candidate metrics are required")
    baseline = validation_summary["baseline"]
    delta = validation_summary["candidate_vs_baseline"]
    candidate_config = config["candidate"]
    checks = {
        "clean_classification_unchanged": (
            delta["clean_accuracy_delta"] == 0 and delta["clean_macro_f1_delta"] == 0
        ),
        "minimum_mean_corruption_macro_f1_improvement": (
            delta["mean_corruption_macro_f1_delta"]
            >= float(candidate_config["minimum_validation_improvement"])
        ),
        "worst_corruption_macro_f1_not_worse": (
            candidate["worst_condition"]["macro_f1"] >= baseline["worst_condition"]["macro_f1"]
        ),
    }
    return {
        "split": config["protocol"]["selection_split"],
        "metric": candidate_config["selection_metric"],
        "minimum_improvement": candidate_config["minimum_validation_improvement"],
        "observed_improvement": delta["mean_corruption_macro_f1_delta"],
        "checks": checks,
        "selected_for_final_test": all(checks.values()),
        "test_metrics_consulted": False,
    }


def evaluate_promotion(
    *,
    selection: dict[str, Any],
    test_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate = test_summary.get("candidate")
    baseline = test_summary["baseline"]
    selected = selection["selected_for_final_test"]
    checks = {
        "candidate_selected_on_validation": selected,
        "test_clean_classification_unchanged": bool(
            candidate
            and candidate["clean"]["accuracy"] == baseline["clean"]["accuracy"]
            and candidate["clean"]["macro_f1"] == baseline["clean"]["macro_f1"]
        ),
        "test_mean_corruption_macro_f1_improved": bool(
            candidate
            and candidate["corruption_average"]["macro_f1"]
            > baseline["corruption_average"]["macro_f1"]
        ),
        "test_worst_corruption_macro_f1_not_worse": bool(
            candidate
            and candidate["worst_condition"]["macro_f1"] >= baseline["worst_condition"]["macro_f1"]
        ),
    }
    evidence_gates_passed = all(checks.values())
    policy_blockers = {
        "synthetic_corruptions_not_proven_production_representative": not config["claim_boundary"][
            "synthetic_corruptions_are_production_representative"
        ],
        "automatic_production_promotion_disabled": not config["promotion_gates"][
            "automatic_production_promotion"
        ],
    }
    return {
        "checks": checks,
        "evidence_gates_passed": evidence_gates_passed,
        "policy_blockers": policy_blockers,
        "production_promotion_approved": (
            evidence_gates_passed and not any(policy_blockers.values())
        ),
    }


def evaluate_split(
    *,
    samples: Sequence[Sample],
    model: nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    include_candidate: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    protocol = config["protocol"]
    for scenario in corruption_scenarios(config):
        dataset = CorruptedImageDataset(
            samples,
            transform,
            corruption=scenario["corruption"],
            parameter=scenario["parameter"],
            severity=scenario["severity"],
            seed=int(protocol["seed"]),
        )
        collected = collect_policy_logits(
            model,
            dataset,
            device,
            batch_size=int(protocol["batch_size"]),
            include_candidate=include_candidate,
            transforms=tuple(config["candidate"]["transforms"]),
        )
        row = {
            **scenario,
            "baseline": calibration_metrics(
                collected["baseline_logits"],
                collected["labels"],
                bin_count=int(protocol["bin_count"]),
            ),
            "candidate": None,
            "baseline_seconds": collected["baseline_seconds"],
            "candidate_seconds": collected["candidate_seconds"],
        }
        if include_candidate:
            row["candidate"] = calibration_metrics(
                collected["candidate_logits"],
                collected["labels"],
                bin_count=int(protocol["bin_count"]),
            )
        rows.append(row)
    return {
        "samples": len(samples),
        "candidate_evaluated": include_candidate,
        "summary": summarize_scenarios(rows),
        "scenarios": rows,
    }


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    title: str,
    summary: dict[str, Any],
) -> None:
    x0, y0 = origin
    width, height = 345, 260
    font = ImageFont.load_default()
    draw.text((x0, y0 - 28), title, fill=(25, 35, 43), font=font)
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(75, 91, 98), width=1)
    for tick in range(6):
        value = tick / 5
        y = y0 + height - int(value * height)
        draw.line((x0, y, x0 + width, y), fill=(218, 217, 210), width=1)
        draw.text((x0 - 32, y - 5), f"{value:.1f}", fill=(75, 91, 98), font=font)
    colors = {"baseline": (232, 108, 77), "candidate": (31, 83, 85)}
    for policy in ("baseline", "candidate"):
        policy_summary = summary.get(policy)
        if policy_summary is None:
            continue
        values = [
            policy_summary["clean"]["macro_f1"],
            *[row["macro_f1"] for row in policy_summary["by_severity"]],
        ]
        points = [
            (
                x0 + int(index * width / 3),
                y0 + height - int(value * height),
            )
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=colors[policy], width=3)
        for point in points:
            draw.ellipse(
                (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                fill=colors[policy],
            )
    for index, label in enumerate(("clean", "S1", "S2", "S3")):
        x = x0 + int(index * width / 3)
        draw.text((x - 10, y0 + height + 8), label, fill=(25, 35, 43), font=font)


def render_robustness_figure(
    validation_summary: dict[str, Any],
    test_summary: dict[str, Any],
    destination: Path,
) -> None:
    canvas = Image.new("RGB", (820, 390), (248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (45, 22),
        "TerraClass corruption robustness: macro F1 by severity",
        fill=(25, 35, 43),
        font=font,
    )
    draw.line((535, 26, 565, 26), fill=(232, 108, 77), width=3)
    draw.text((571, 21), "single-view", fill=(25, 35, 43), font=font)
    draw.line((660, 26, 690, 26), fill=(31, 83, 85), width=3)
    draw.text((696, 21), "dihedral-4 TTA", fill=(25, 35, 43), font=font)
    _draw_panel(draw, (45, 75), "Validation (candidate selection)", validation_summary)
    test_title = (
        "Test (single-view only; TTA rejected)"
        if test_summary.get("candidate") is None
        else "Test (opened after selection)"
    )
    _draw_panel(draw, (440, 75), test_title, test_summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": Image.__version__,
    }


def run_robustness_evaluation(
    *,
    project_root: Path,
    config_path: Path,
    output_path: Path,
    figure_path: Path,
    device: str,
    report_date: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_robustness_config(config_path)
    serving = load_serving_config(root / config["serving_config_path"])
    predictor = TerraClassPredictor.load(serving, root, device=device)
    baseline = load_config(root / serving.baseline_config_path)
    dataset_root = root / config["dataset_root"]
    splits, _ = load_manifest(
        root / serving.training_manifest_path,
        dataset_root,
        baseline,
        verify_hashes=True,
    )

    validation = evaluate_split(
        samples=splits[config["protocol"]["selection_split"]],
        model=predictor.model,
        transform=predictor.transform,
        device=predictor.device,
        config=config,
        include_candidate=True,
    )
    selection = evaluate_candidate_selection(validation["summary"], config)
    test = evaluate_split(
        samples=splits[config["protocol"]["evaluation_split"]],
        model=predictor.model,
        transform=predictor.transform,
        device=predictor.device,
        config=config,
        include_candidate=bool(selection["selected_for_final_test"]),
    )
    promotion = evaluate_promotion(
        selection=selection,
        test_summary=test["summary"],
        config=config,
    )
    render_robustness_figure(
        validation["summary"],
        test["summary"],
        figure_path,
    )
    report = {
        "schema_version": 1,
        "evaluated_on": report_date,
        "phase": {
            "scheduled_date": "2026-07-20",
            "status": "completed_early",
        },
        "evaluation_id": config["evaluation_id"],
        "environment": _environment(),
        "model": {
            "model_id": serving.model_id,
            "model_version": serving.model_version,
            "architecture": serving.architecture,
            "class_names": list(serving.class_names),
            "serving_artifact_sha256": serving.serving_artifact.sha256,
        },
        "provenance": {
            "config_path": config_path.resolve().relative_to(root).as_posix(),
            "config_sha256": file_sha256(config_path),
            "manifest_path": serving.training_manifest_path,
            "manifest_sha256": serving.training_manifest_sha256,
            "dataset_archive_sha256": baseline.dataset.archive_sha256,
            "validation_samples": len(splits["validation"]),
            "test_samples": len(splits["test"]),
        },
        "protocol": {
            **config["protocol"],
            "scenario_count_per_split": len(corruption_scenarios(config)),
            "candidate": config["candidate"],
            "selection_precedes_test": True,
        },
        "validation": validation,
        "candidate_selection": selection,
        "test": test,
        "promotion": promotion,
        "methodology_references": config["methodology_references"],
        "figure": {
            "path": figure_path.resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(figure_path),
        },
        "claim_boundary": {
            **config["claim_boundary"],
            "production_model_changed": False,
            "production_serving_policy_changed": False,
            "calibration_policy_changed": False,
            "candidate_is_evaluation_only": True,
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
        default=Path("configs/evaluation/robustness_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/robustness_evaluation_2026-07-18.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/robustness_resnet18_group_aware_2026-07-18.png"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--report-date", default="2026-07-18")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    report = run_robustness_evaluation(
        project_root=root,
        config_path=rooted(args.config),
        output_path=rooted(args.output),
        figure_path=rooted(args.figure),
        device=args.device,
        report_date=args.report_date,
    )
    validation = report["validation"]["summary"]
    test = report["test"]["summary"]
    print(
        json.dumps(
            {
                "candidate_selected": report["candidate_selection"]["selected_for_final_test"],
                "validation_mean_macro_f1": {
                    "baseline": validation["baseline"]["corruption_average"]["macro_f1"],
                    "candidate": validation["candidate"]["corruption_average"]["macro_f1"],
                },
                "test_mean_macro_f1": {
                    "baseline": test["baseline"]["corruption_average"]["macro_f1"],
                    "candidate": (
                        test["candidate"]["corruption_average"]["macro_f1"]
                        if test["candidate"]
                        else None
                    ),
                },
                "production_promotion_approved": report["promotion"][
                    "production_promotion_approved"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
