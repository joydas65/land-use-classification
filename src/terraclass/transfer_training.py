"""Run two-stage transfer learning on historical or group-aware manifests."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn, optim

from terraclass.config import ExperimentConfig, load_config
from terraclass.data import ImagePathDataset, file_sha256, load_manifest
from terraclass.devices import select_device
from terraclass.training import build_loader, set_reproducibility, train_one_epoch
from terraclass.transfer import (
    build_transfer_model,
    set_backbone_trainable,
    trainable_parameter_count,
)
from terraclass.transfer_config import TransferConfig, load_transfer_config
from terraclass.transforms import build_eval_transform, build_train_transform


def evaluate_transfer(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate with class-balanced and top-k metrics needed for model comparison."""
    model.eval()
    total_loss = 0.0
    total = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    top3_correct = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            total += labels.size(0)
            predictions = outputs.argmax(dim=1)
            top3 = outputs.topk(min(3, len(class_names)), dim=1).indices
            top3_correct += top3.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    labels = list(range(len(class_names)))
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "top3_accuracy": top3_correct / total,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
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


def run_transfer_experiment(
    dataset_root: Path,
    baseline: ExperimentConfig,
    transfer: TransferConfig,
    output_dir: Path,
    device: torch.device,
    *,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train the head, fine-tune all layers, and select by validation macro-F1."""
    set_reproducibility(baseline.seed)
    manifest = Path(transfer.manifest_path)
    manifest_hash = file_sha256(manifest)
    if manifest_hash != transfer.manifest_sha256:
        raise RuntimeError(
            f"Manifest hash {manifest_hash} differs from configured hash {transfer.manifest_sha256}"
        )
    splits, _ = load_manifest(manifest, dataset_root, baseline, verify_hashes=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ImagePathDataset(splits["train"], build_train_transform(baseline.preprocessing))
    validation_dataset = ImagePathDataset(
        splits["validation"], build_eval_transform(baseline.preprocessing)
    )
    test_dataset = ImagePathDataset(splits["test"], build_eval_transform(baseline.preprocessing))
    loader_kwargs = {
        "config": baseline,
        "num_workers": num_workers,
        "batch_size": transfer.batch_size,
    }
    train_loader = build_loader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = build_loader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = build_loader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_transfer_model(
        transfer.architecture,
        baseline.class_count,
        pretrained=transfer.pretrained,
        dropout=transfer.dropout,
    ).to(device)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = output_dir / "best_model.pth"
    history: list[dict[str, Any]] = []
    best_score = -1.0
    selected: dict[str, Any] = {}
    global_epoch = 0

    stages = (
        ("head", transfer.head_epochs, transfer.head_learning_rate, False),
        ("fine_tune", transfer.fine_tune_epochs, transfer.fine_tune_learning_rate, True),
    )
    for stage_name, max_epochs, learning_rate, backbone_trainable in stages:
        set_backbone_trainable(model, transfer.architecture, backbone_trainable)
        optimizer = optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=transfer.weight_decay,
        )
        epochs_without_improvement = 0
        for stage_epoch in range(1, max_epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            validation_metrics = evaluate_transfer(
                model,
                validation_loader,
                criterion,
                device,
                baseline.dataset.selected_classes,
            )
            history.append(
                {
                    "global_epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch,
                    "learning_rate": learning_rate,
                    "trainable_parameters": trainable_parameter_count(model),
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            score = float(validation_metrics[transfer.selection_metric])
            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                selected = {
                    "global_epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch,
                    "validation_macro_f1": score,
                }
                torch.save(
                    {
                        "schema_version": 1,
                        "model_state_dict": model.state_dict(),
                        "class_names": baseline.dataset.selected_classes,
                        "baseline_config": asdict(baseline),
                        "transfer_config": asdict(transfer),
                        "manifest_sha256": manifest_hash,
                        "selected": selected,
                        "environment": _environment(),
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= transfer.early_stopping_patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_transfer(
        model, test_loader, criterion, device, baseline.dataset.selected_classes
    )
    result = {
        "schema_version": 1,
        "experiment_name": transfer.experiment_name,
        "seed": baseline.seed,
        "device": str(device),
        "environment": _environment(),
        "architecture": transfer.architecture,
        "pretrained": transfer.pretrained,
        "split_kind": transfer.split_kind,
        "split_counts": {name: len(items) for name, items in splits.items()},
        "split_manifest_sha256": manifest_hash,
        "total_parameters": total_parameter_count,
        "selection_metric": transfer.selection_metric,
        "selected": selected,
        "history": history,
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-cache", type=Path, default=Path(".cache/torch"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights_cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(args.weights_cache.resolve()))
    transfer = load_transfer_config(args.config)
    baseline = load_config(transfer.baseline_config_path)
    result = run_transfer_experiment(
        args.dataset_root,
        baseline,
        transfer,
        args.output_dir,
        select_device(args.device),
        num_workers=args.num_workers,
    )
    print(json.dumps({"selected": result["selected"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
