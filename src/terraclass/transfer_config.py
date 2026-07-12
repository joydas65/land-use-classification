"""Configuration schema for comparable transfer-learning experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terraclass.transfer import SUPPORTED_ARCHITECTURES


@dataclass(frozen=True)
class TransferConfig:
    schema_version: int
    experiment_name: str
    baseline_config_path: str
    split_kind: str
    manifest_path: str
    manifest_sha256: str
    architecture: str
    pretrained: bool
    dropout: float
    batch_size: int
    head_epochs: int
    fine_tune_epochs: int
    head_learning_rate: float
    fine_tune_learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    selection_metric: str

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.split_kind not in {"historical", "group_aware"}:
            errors.append("split_kind must be historical or group_aware")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            errors.append(f"unsupported architecture: {self.architecture}")
        if len(self.manifest_sha256) != 64:
            errors.append("manifest_sha256 must contain 64 hexadecimal characters")
        if not 0 <= self.dropout < 1:
            errors.append("dropout must be in [0, 1)")
        if min(self.batch_size, self.head_epochs, self.fine_tune_epochs) <= 0:
            errors.append("batch_size and both epoch counts must be positive")
        if min(self.head_learning_rate, self.fine_tune_learning_rate, self.weight_decay) <= 0:
            errors.append("learning rates and weight_decay must be positive")
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        if self.selection_metric != "macro_f1":
            errors.append("selection_metric must be macro_f1")
        if errors:
            raise ValueError("Invalid transfer configuration: " + "; ".join(errors))


def transfer_config_from_dict(raw: dict[str, Any]) -> TransferConfig:
    model = raw["model"]
    training = raw["training"]
    config = TransferConfig(
        schema_version=int(raw["schema_version"]),
        experiment_name=str(raw["experiment_name"]),
        baseline_config_path=str(raw["baseline_config_path"]),
        split_kind=str(raw["split_kind"]),
        manifest_path=str(raw["manifest_path"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        architecture=str(model["architecture"]),
        pretrained=bool(model["pretrained"]),
        dropout=float(model["dropout"]),
        batch_size=int(training["batch_size"]),
        head_epochs=int(training["head_epochs"]),
        fine_tune_epochs=int(training["fine_tune_epochs"]),
        head_learning_rate=float(training["head_learning_rate"]),
        fine_tune_learning_rate=float(training["fine_tune_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        selection_metric=str(raw["selection_metric"]),
    )
    config.validate()
    return config


def load_transfer_config(path: str | Path) -> TransferConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return transfer_config_from_dict(json.load(handle))
