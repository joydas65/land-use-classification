import numpy as np
import torch
from PIL import Image

from terraclass.config import ExperimentConfig
from terraclass.transforms import build_eval_transform, build_train_transform


def test_eval_transform_has_expected_shape_and_is_deterministic(
    baseline_config: ExperimentConfig,
) -> None:
    image = Image.fromarray(np.full((32, 40, 3), 128, dtype=np.uint8))
    transform = build_eval_transform(baseline_config.preprocessing)
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_training_transform_has_expected_shape(baseline_config: ExperimentConfig) -> None:
    image = Image.fromarray(np.full((32, 40, 3), 128, dtype=np.uint8))
    transformed = build_train_transform(baseline_config.preprocessing)(image)
    assert transformed.shape == (3, 224, 224)
