"""Reproducible training entry point for the original five-class baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from terraclass.config import ExperimentConfig, load_config
from terraclass.data import (
    ImagePathDataset,
    discover_samples,
    file_sha256,
    stratified_split,
    write_manifest,
)
from terraclass.model import LandUseCNN, count_trainable_parameters
from terraclass.transforms import build_eval_transform, build_train_transform


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset: ImagePathDataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    num_workers: int,
    batch_size: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("Cannot train on an empty data loader")
    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().tolist())
    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    label_indices = list(range(len(class_names)))
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=label_indices
        ).tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=label_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_baseline(
    dataset_root: Path,
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, Any]:
    set_reproducibility(config.seed)
    samples = discover_samples(dataset_root, config)
    splits = stratified_split(samples, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = output_dir / "split_manifest.csv"
    write_manifest(run_manifest, splits, dataset_root, include_hashes=True)
    manifest_hash = file_sha256(run_manifest)
    if manifest_hash != config.split.manifest_sha256:
        raise RuntimeError(
            f"Generated manifest hash {manifest_hash} differs from the configured baseline "
            f"hash {config.split.manifest_sha256}"
        )

    train_dataset = ImagePathDataset(splits["train"], build_train_transform(config.preprocessing))
    validation_dataset = ImagePathDataset(
        splits["validation"], build_eval_transform(config.preprocessing)
    )
    test_dataset = ImagePathDataset(splits["test"], build_eval_transform(config.preprocessing))
    train_loader = build_loader(train_dataset, config, shuffle=True, num_workers=num_workers)
    validation_loader = build_loader(
        validation_dataset, config, shuffle=False, num_workers=num_workers
    )
    test_loader = build_loader(test_dataset, config, shuffle=False, num_workers=num_workers)

    model = LandUseCNN(config.class_count).to(device)
    if count_trainable_parameters(model) != config.observed_baseline.parameter_count:
        raise RuntimeError("Model parameter count differs from the accepted baseline")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.scheduler_step_size,
        gamma=config.training.scheduler_gamma,
    )

    history: list[dict[str, Any]] = []
    best_validation_accuracy = -1.0
    checkpoint_path = output_dir / "best_baseline_model.pth"
    for epoch in range(1, config.training.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate(
            model, validation_loader, criterion, device, config.dataset.selected_classes
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            torch.save(
                {
                    "schema_version": 1,
                    "model_state_dict": model.state_dict(),
                    "class_names": config.dataset.selected_classes,
                    "config": asdict(config),
                    "epoch": epoch,
                    "validation_accuracy": best_validation_accuracy,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, config.dataset.selected_classes)
    result = {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "device": str(device),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_manifest_sha256": manifest_hash,
        "history": history,
        "selected_epoch": checkpoint["epoch"],
        "test": test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_5class.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_5class"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_baseline(
        args.dataset_root,
        config,
        args.output_dir,
        select_device(args.device),
        args.num_workers,
    )
    print(
        json.dumps({"test": result["test"], "selected_epoch": result["selected_epoch"]}, indent=2)
    )


if __name__ == "__main__":
    main()
