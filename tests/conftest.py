from pathlib import Path

import pytest

from terraclass.config import ExperimentConfig, load_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def baseline_config(project_root: Path) -> ExperimentConfig:
    return load_config(project_root / "configs/baseline_5class.json")
