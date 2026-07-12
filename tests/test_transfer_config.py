from pathlib import Path

import pytest

from terraclass.transfer_config import load_transfer_config, transfer_config_from_dict


def test_all_transfer_configs_are_valid(project_root: Path) -> None:
    paths = sorted((project_root / "configs/transfer").glob("*.json"))
    assert len(paths) == 4
    configs = [load_transfer_config(path) for path in paths]
    assert {config.architecture for config in configs} == {"resnet18", "efficientnet_b0"}
    assert {config.split_kind for config in configs} == {"historical", "group_aware"}


def test_transfer_config_rejects_accuracy_selection() -> None:
    raw = {
        "schema_version": 1,
        "experiment_name": "bad",
        "baseline_config_path": "configs/baseline_5class.json",
        "split_kind": "historical",
        "manifest_path": "manifest.csv",
        "manifest_sha256": "a" * 64,
        "model": {"architecture": "resnet18", "pretrained": True, "dropout": 0.2},
        "training": {
            "batch_size": 16,
            "head_epochs": 3,
            "fine_tune_epochs": 7,
            "head_learning_rate": 0.001,
            "fine_tune_learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 3,
        },
        "selection_metric": "accuracy",
    }
    with pytest.raises(ValueError, match="selection_metric"):
        transfer_config_from_dict(raw)
