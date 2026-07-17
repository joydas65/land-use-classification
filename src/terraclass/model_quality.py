"""Post-hoc calibration, uncertainty, and Grad-CAM evaluation for TerraClass."""

from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from terraclass.config import load_config
from terraclass.data import ImagePathDataset, Sample, file_sha256, load_manifest
from terraclass.inference import TerraClassPredictor, load_serving_config


@dataclass(frozen=True)
class CalibrationConfig:
    fit_split: str
    evaluation_split: str
    bin_count: int
    temperature_bounds: tuple[float, float]
    optimization_iterations: int


@dataclass(frozen=True)
class ExplainabilityConfig:
    method: str
    target_layer: str
    selection_policy: str
    samples_per_class: int


@dataclass(frozen=True)
class ModelQualityConfig:
    schema_version: int
    evaluation_id: str
    serving_config_path: str
    dataset_root: str
    calibration: CalibrationConfig
    confidence_thresholds: tuple[float, ...]
    explainability: ExplainabilityConfig

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if not self.evaluation_id.strip():
            errors.append("evaluation_id must not be blank")
        for field, value in (
            ("serving_config_path", self.serving_config_path),
            ("dataset_root", self.dataset_root),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"{field} must be project-relative")
        if self.calibration.fit_split != "validation":
            errors.append("temperature fitting must use the validation split")
        if self.calibration.evaluation_split != "test":
            errors.append("calibration evaluation must use the untouched test split")
        if self.calibration.bin_count < 2:
            errors.append("bin_count must be at least 2")
        lower, upper = self.calibration.temperature_bounds
        if not 0 < lower < upper:
            errors.append("temperature_bounds must be positive and increasing")
        if self.calibration.optimization_iterations < 16:
            errors.append("optimization_iterations must be at least 16")
        if (
            not self.confidence_thresholds
            or tuple(sorted(set(self.confidence_thresholds))) != self.confidence_thresholds
            or any(not 0 <= value <= 1 for value in self.confidence_thresholds)
        ):
            errors.append("confidence_thresholds must be unique, sorted, and within [0, 1]")
        if self.explainability.method != "grad_cam":
            errors.append("explainability method must be grad_cam")
        if self.explainability.target_layer != "layer4[-1]":
            errors.append("the selected ResNet18 target layer must be layer4[-1]")
        if self.explainability.selection_policy != "lexicographically_first_test_sample_per_class":
            errors.append("explainability samples must use the deterministic selection policy")
        if self.explainability.samples_per_class != 1:
            errors.append("exactly one explainability sample per class is required")
        if errors:
            raise ValueError("Invalid model-quality configuration: " + "; ".join(errors))


def load_model_quality_config(path: str | Path) -> ModelQualityConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    calibration = raw["calibration"]
    explainability = raw["explainability"]
    config = ModelQualityConfig(
        schema_version=int(raw["schema_version"]),
        evaluation_id=str(raw["evaluation_id"]),
        serving_config_path=str(raw["serving_config_path"]),
        dataset_root=str(raw["dataset_root"]),
        calibration=CalibrationConfig(
            fit_split=str(calibration["fit_split"]),
            evaluation_split=str(calibration["evaluation_split"]),
            bin_count=int(calibration["bin_count"]),
            temperature_bounds=tuple(float(value) for value in calibration["temperature_bounds"]),
            optimization_iterations=int(calibration["optimization_iterations"]),
        ),
        confidence_thresholds=tuple(
            float(value) for value in raw["selective_prediction"]["confidence_thresholds"]
        ),
        explainability=ExplainabilityConfig(
            method=str(explainability["method"]),
            target_layer=str(explainability["target_layer"]),
            selection_policy=str(explainability["selection_policy"]),
            samples_per_class=int(explainability["samples_per_class"]),
        ),
    )
    config.validate()
    return config


def _validate_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [non-empty samples, at least two classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have one entry per logit row")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("labels must use an integer dtype")
    if int(labels.min()) < 0 or int(labels.max()) >= logits.shape[1]:
        raise ValueError("labels must be valid class indices")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")


def reliability_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bin_count: int,
) -> list[dict[str, int | float | None]]:
    """Summarize fixed-width top-one calibration bins."""
    _validate_logits_labels(logits, labels)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    probabilities = torch.softmax(logits.double(), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    bins: list[dict[str, int | float | None]] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if index == bin_count - 1:
            selected = confidence.ge(lower) & confidence.le(upper)
        else:
            selected = confidence.ge(lower) & confidence.lt(upper)
        count = int(selected.sum())
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": float(confidence[selected].mean()) if count else None,
                "accuracy": float(correct[selected].double().mean()) if count else None,
            }
        )
    return bins


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bin_count: int,
) -> dict[str, Any]:
    """Calculate top-one ECE, multiclass Brier score, NLL, and uncertainty summaries."""
    _validate_logits_labels(logits, labels)
    probabilities = torch.softmax(logits.double(), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    bins = reliability_bins(logits, labels, bin_count=bin_count)
    sample_count = len(labels)
    ece = sum(
        int(item["count"])
        / sample_count
        * abs(float(item["accuracy"]) - float(item["mean_confidence"]))
        for item in bins
        if item["count"]
    )
    one_hot = functional.one_hot(labels.long(), num_classes=logits.shape[1]).double()
    entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(logits.shape[1])
    confidence_values = confidence.detach().cpu().numpy()
    return {
        "sample_count": sample_count,
        "accuracy": float(correct.double().mean()),
        "macro_f1": float(
            f1_score(
                labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": float(functional.cross_entropy(logits.double(), labels.long())),
        "multiclass_brier_score": float(((probabilities - one_hot) ** 2).sum(dim=1).mean()),
        "expected_calibration_error": float(ece),
        "mean_confidence": float(confidence.mean()),
        "minimum_confidence": float(confidence.min()),
        "confidence_quantiles": {
            "p25": float(np.quantile(confidence_values, 0.25)),
            "p50": float(np.quantile(confidence_values, 0.50)),
            "p75": float(np.quantile(confidence_values, 0.75)),
        },
        "mean_normalized_predictive_entropy": float(normalized_entropy.mean()),
        "reliability_bins": bins,
    }


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bounds: tuple[float, float],
    iterations: int,
) -> dict[str, float | bool]:
    """Fit one positive temperature by validation NLL using bounded golden-section search."""
    _validate_logits_labels(logits, labels)
    lower, upper = bounds
    if not 0 < lower < upper:
        raise ValueError("temperature bounds must be positive and increasing")
    if iterations < 16:
        raise ValueError("iterations must be at least 16")
    logits = logits.detach().double()
    labels = labels.detach().long()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(functional.cross_entropy(logits / temperature, labels))

    log_lower, log_upper = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5) - 1) / 2
    left = log_upper - ratio * (log_upper - log_lower)
    right = log_lower + ratio * (log_upper - log_lower)
    left_value, right_value = objective(left), objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            log_upper, right, right_value = right, left, left_value
            left = log_upper - ratio * (log_upper - log_lower)
            left_value = objective(left)
        else:
            log_lower, left, left_value = left, right, right_value
            right = log_lower + ratio * (log_upper - log_lower)
            right_value = objective(right)
    candidates = (
        (bounds[0], objective(math.log(bounds[0]))),
        (math.exp((log_lower + log_upper) / 2), objective((log_lower + log_upper) / 2)),
        (bounds[1], objective(math.log(bounds[1]))),
    )
    temperature, fitted_nll = min(candidates, key=lambda item: item[1])
    tolerance = max(1e-8, bounds[0] * 1e-6)
    at_lower = abs(temperature - bounds[0]) <= tolerance
    at_upper = abs(temperature - bounds[1]) <= max(1e-8, bounds[1] * 1e-6)
    return {
        "temperature": float(temperature),
        "validation_nll_before": objective(0.0),
        "validation_nll_after": fitted_nll,
        "at_optimization_bound": at_lower or at_upper,
        "at_lower_bound": at_lower,
        "at_upper_bound": at_upper,
    }


def selective_prediction_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: tuple[float, ...],
) -> list[dict[str, int | float | None]]:
    """Report retained coverage and risk when abstaining below confidence thresholds."""
    _validate_logits_labels(logits, labels)
    if tuple(sorted(set(thresholds))) != thresholds or any(
        not 0 <= value <= 1 for value in thresholds
    ):
        raise ValueError("thresholds must be unique, sorted, and within [0, 1]")
    probabilities = torch.softmax(logits.double(), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    rows: list[dict[str, int | float | None]] = []
    for threshold in thresholds:
        retained = confidence.ge(threshold)
        count = int(retained.sum())
        accuracy = float(correct[retained].double().mean()) if count else None
        rows.append(
            {
                "confidence_threshold": threshold,
                "retained_count": count,
                "coverage": count / len(labels),
                "accuracy": accuracy,
                "selective_risk": None if accuracy is None else 1 - accuracy,
            }
        )
    return rows


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    *,
    class_index: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute a normalized Grad-CAM heatmap for one image tensor."""
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [1, channels, height, width]")
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(gradients.append)

    handle = target_layer.register_forward_hook(capture_activation)
    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model output must have shape [1, classes]")
        predicted = int(logits.argmax(dim=1))
        target = predicted if class_index is None else class_index
        if not 0 <= target < logits.shape[1]:
            raise ValueError("class_index is outside the model output range")
        logits[0, target].backward()
    finally:
        handle.remove()
        model.train(was_training)
    if len(activations) != 1 or len(gradients) != 1:
        raise RuntimeError("target layer must run exactly once during Grad-CAM")
    activation, gradient = activations[0], gradients[0]
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise RuntimeError("Grad-CAM target layer must produce a four-dimensional feature map")
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
    heatmap = functional.interpolate(
        heatmap,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    minimum, maximum = heatmap.min(), heatmap.max()
    if float((maximum - minimum).detach()) > 0:
        heatmap = (heatmap - minimum) / (maximum - minimum)
    else:
        heatmap = torch.zeros_like(heatmap)
    probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
    return heatmap.detach().cpu().numpy(), predicted, probabilities


def deterministic_explainability_samples(
    samples: list[Sample],
    class_names: tuple[str, ...],
) -> list[Sample]:
    """Select the lexicographically first test sample for every configured class."""
    chosen: list[Sample] = []
    for class_name in class_names:
        candidates = sorted(
            (sample for sample in samples if sample.class_name == class_name),
            key=lambda sample: sample.path.as_posix(),
        )
        if not candidates:
            raise ValueError(f"No explainability sample exists for class {class_name}")
        chosen.append(candidates[0])
    return chosen


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("heatmap must be a finite two-dimensional array")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    values = np.clip(heatmap, 0, 1)
    red = np.clip(1.5 * values, 0, 1)
    green = np.clip(1.5 - np.abs(2 * values - 1.0) * 1.5, 0, 1)
    blue = np.clip(1.5 * (1 - values), 0, 1)
    colored = Image.fromarray(
        np.uint8(np.stack((red, green, blue), axis=-1) * 255),
        mode="RGB",
    )
    return Image.blend(base, colored, alpha)


def _collect_logits(
    model: nn.Module,
    dataset: ImagePathDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(logits), torch.cat(labels)


def _draw_reliability_panel(
    canvas: Image.Image,
    origin: tuple[int, int],
    title: str,
    bins: list[dict[str, int | float | None]],
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x, y = origin
    width, height = 360, 360
    draw.text((x, y - 28), title, fill=(25, 35, 43), font=font)
    draw.rectangle((x, y, x + width, y + height), outline=(74, 91, 98), width=2)
    draw.line((x, y + height, x + width, y), fill=(120, 130, 135), width=2)
    bin_width = width / len(bins)
    for index, item in enumerate(bins):
        if not item["count"]:
            continue
        accuracy = float(item["accuracy"])
        confidence = float(item["mean_confidence"])
        left = int(x + index * bin_width + 3)
        right = int(x + (index + 1) * bin_width - 3)
        accuracy_top = int(y + height * (1 - accuracy))
        confidence_y = int(y + height * (1 - confidence))
        draw.rectangle(
            (left, accuracy_top, right, y + height),
            fill=(231, 111, 81),
            outline=(174, 74, 53),
        )
        draw.line((left, confidence_y, right, confidence_y), fill=(22, 68, 74), width=3)
    draw.text((x + 115, y + height + 12), "Top-one confidence", fill=(25, 35, 43), font=font)
    draw.text((x + 8, y + 8), "Accuracy bars / confidence lines", fill=(25, 35, 43), font=font)


def render_reliability_figure(
    before: dict[str, Any],
    after: dict[str, Any],
    destination: Path,
) -> None:
    canvas = Image.new("RGB", (860, 500), (248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (45, 25),
        "ResNet18 group-aware test reliability (75 images)",
        fill=(25, 35, 43),
        font=font,
    )
    _draw_reliability_panel(canvas, (45, 80), "Original softmax", before["reliability_bins"])
    _draw_reliability_panel(
        canvas, (455, 80), "Temperature-scaling sensitivity", after["reliability_bins"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def render_gradcam_figure(
    rows: list[dict[str, Any]],
    originals: list[Image.Image],
    overlays: list[Image.Image],
    destination: Path,
) -> None:
    tile_width, image_size = 280, 256
    canvas = Image.new("RGB", (tile_width * len(rows), 650), (248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (20, 15),
        "Deterministic Grad-CAM review — original images (top), overlays (bottom)",
        fill=(25, 35, 43),
        font=font,
    )
    for index, (row, original, overlay) in enumerate(zip(rows, originals, overlays, strict=True)):
        x = index * tile_width + 12
        canvas.paste(original.resize((image_size, image_size)), (x, 55))
        canvas.paste(overlay.resize((image_size, image_size)), (x, 340))
        label = f"{row['true_class']} -> {row['predicted_class']} ({float(row['confidence']):.3f})"
        draw.text((x, 320), label, fill=(25, 35, 43), font=font)
        draw.text((x, 605), Path(row["relative_path"]).name, fill=(25, 35, 43), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_model_quality_evaluation(
    *,
    project_root: Path,
    config_path: Path,
    output_path: Path,
    reliability_figure_path: Path,
    gradcam_figure_path: Path,
    report_date: str,
    device: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_model_quality_config(config_path)
    serving_config = load_serving_config(root / config.serving_config_path)
    predictor = TerraClassPredictor.load(serving_config, root, device=device)
    baseline = load_config(root / serving_config.baseline_config_path)
    splits, _ = load_manifest(
        root / serving_config.training_manifest_path,
        root / config.dataset_root,
        baseline,
        verify_hashes=True,
    )
    validation_dataset = ImagePathDataset(splits["validation"], predictor.transform)
    test_dataset = ImagePathDataset(splits["test"], predictor.transform)
    validation_logits, validation_labels = _collect_logits(
        predictor.model, validation_dataset, predictor.device
    )
    test_logits, test_labels = _collect_logits(predictor.model, test_dataset, predictor.device)
    fitted = fit_temperature(
        validation_logits,
        validation_labels,
        bounds=config.calibration.temperature_bounds,
        iterations=config.calibration.optimization_iterations,
    )
    temperature = float(fitted["temperature"])
    validation_before = calibration_metrics(
        validation_logits,
        validation_labels,
        bin_count=config.calibration.bin_count,
    )
    validation_after = calibration_metrics(
        validation_logits / temperature,
        validation_labels,
        bin_count=config.calibration.bin_count,
    )
    test_before = calibration_metrics(
        test_logits,
        test_labels,
        bin_count=config.calibration.bin_count,
    )
    test_after = calibration_metrics(
        test_logits / temperature,
        test_labels,
        bin_count=config.calibration.bin_count,
    )
    calibration_reliable = not bool(fitted["at_optimization_bound"])
    render_reliability_figure(test_before, test_after, reliability_figure_path)

    chosen = deterministic_explainability_samples(splits["test"], serving_config.class_names)
    gradcam_rows: list[dict[str, Any]] = []
    originals: list[Image.Image] = []
    overlays: list[Image.Image] = []
    if serving_config.architecture != "resnet18":
        raise ValueError("This evaluation contract requires the selected ResNet18 architecture")
    target_layer = predictor.model.layer4[-1]
    dataset_root = (root / config.dataset_root).resolve()
    for sample in chosen:
        with Image.open(sample.path) as source:
            original = source.convert("RGB")
        tensor = predictor.transform(original).unsqueeze(0).to(predictor.device)
        heatmap, predicted, probabilities = compute_gradcam(
            predictor.model,
            tensor,
            target_layer,
        )
        overlay = overlay_gradcam(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap)
        row = {
            "relative_path": sample.path.resolve().relative_to(dataset_root).as_posix(),
            "image_sha256": file_sha256(sample.path),
            "true_class": sample.class_name,
            "predicted_class": serving_config.class_names[predicted],
            "confidence": float(probabilities[predicted]),
            "correct": predicted == sample.label,
            "target_class": serving_config.class_names[predicted],
        }
        gradcam_rows.append(row)
        originals.append(original)
        overlays.append(overlay)
    render_gradcam_figure(gradcam_rows, originals, overlays, gradcam_figure_path)

    report = {
        "schema_version": 1,
        "evaluated_on": report_date,
        "phase": {
            "scheduled_date": "2026-07-19",
            "status": "completed_early",
        },
        "evaluation_id": config.evaluation_id,
        "environment": _environment(),
        "model": {
            "model_id": serving_config.model_id,
            "model_version": serving_config.model_version,
            "architecture": serving_config.architecture,
            "class_names": list(serving_config.class_names),
        },
        "provenance": {
            "evaluation_config_path": config_path.resolve().relative_to(root).as_posix(),
            "evaluation_config_sha256": file_sha256(config_path),
            "serving_config_path": config.serving_config_path,
            "serving_artifact_sha256": serving_config.serving_artifact.sha256,
            "source_checkpoint_sha256": serving_config.source_checkpoint.sha256,
            "manifest_path": serving_config.training_manifest_path,
            "manifest_sha256": serving_config.training_manifest_sha256,
        },
        "protocol": {
            "calibration_fit_split": config.calibration.fit_split,
            "calibration_fit_samples": len(validation_labels),
            "evaluation_split": config.calibration.evaluation_split,
            "evaluation_samples": len(test_labels),
            "bin_count": config.calibration.bin_count,
            "temperature_bounds": list(config.calibration.temperature_bounds),
            "selective_prediction_thresholds": list(config.confidence_thresholds),
            "explainability_selection_policy": config.explainability.selection_policy,
            "explainability_samples_per_class": config.explainability.samples_per_class,
        },
        "temperature_scaling": {
            "method": "single_scalar_validation_nll",
            "fit": fitted,
            "calibration_fit_reliable": calibration_reliable,
            "deployment_approved": calibration_reliable,
            "deployment_decision": (
                "eligible_for_separate_serving-validation review"
                if calibration_reliable
                else "retain original softmax; fitted temperature reached an optimization bound"
            ),
            "validation": {
                "before": validation_before,
                "after_sensitivity": validation_after,
            },
            "test": {
                "before": test_before,
                "after_sensitivity": test_after,
            },
        },
        "uncertainty": {
            "method": "top_one_confidence_predictive_entropy_and_selective_risk",
            "original_softmax_selective_prediction": selective_prediction_curve(
                test_logits,
                test_labels,
                config.confidence_thresholds,
            ),
            "temperature_sensitivity_selective_prediction": selective_prediction_curve(
                test_logits / temperature,
                test_labels,
                config.confidence_thresholds,
            ),
        },
        "explainability": {
            "method": "grad_cam",
            "target_layer": config.explainability.target_layer,
            "target_policy": "model_predicted_class",
            "selection_policy": config.explainability.selection_policy,
            "samples": gradcam_rows,
        },
        "figures": {
            "reliability": {
                "path": reliability_figure_path.resolve().relative_to(root).as_posix(),
                "sha256": file_sha256(reliability_figure_path),
            },
            "gradcam": {
                "path": gradcam_figure_path.resolve().relative_to(root).as_posix(),
                "sha256": file_sha256(gradcam_figure_path),
            },
        },
        "claim_boundary": {
            "dataset_scope": "balanced 500-image, five-class UC Merced subset",
            "validation_samples": len(validation_labels),
            "test_samples": len(test_labels),
            "calibration_deployed": False,
            "calibration_reason": (
                "A 75-image perfectly classified validation split cannot identify a stable "
                "temperature when the bounded optimum is reached."
                if not calibration_reliable
                else (
                    "This offline result has not been promoted through the serving release process."
                )
            ),
            "gradcam_interpretation": (
                "Grad-CAM is a qualitative localization aid, not a causal explanation or proof "
                "that the model will generalize."
            ),
            "iit_submission_changed": False,
            "production_model_changed": False,
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
        default=Path("configs/evaluation/model_quality_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_quality_evaluation_2026-07-17.json"),
    )
    parser.add_argument(
        "--reliability-figure",
        type=Path,
        default=Path("reports/figures/reliability_resnet18_group_aware_2026-07-17.png"),
    )
    parser.add_argument(
        "--gradcam-figure",
        type=Path,
        default=Path("reports/figures/gradcam_resnet18_group_aware_2026-07-17.png"),
    )
    parser.add_argument("--report-date", default="2026-07-17")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    report = run_model_quality_evaluation(
        project_root=root,
        config_path=rooted(args.config),
        output_path=rooted(args.output),
        reliability_figure_path=rooted(args.reliability_figure),
        gradcam_figure_path=rooted(args.gradcam_figure),
        report_date=args.report_date,
        device=args.device,
    )
    summary = {
        "temperature": report["temperature_scaling"]["fit"]["temperature"],
        "calibration_fit_reliable": report["temperature_scaling"]["calibration_fit_reliable"],
        "test_before": report["temperature_scaling"]["test"]["before"],
        "test_after_sensitivity": report["temperature_scaling"]["test"]["after_sensitivity"],
        "gradcam_samples": len(report["explainability"]["samples"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
