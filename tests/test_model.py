import pytest
import torch

from terraclass.model import LandUseCNN, count_trainable_parameters


def test_baseline_model_shape_and_parameter_count() -> None:
    model = LandUseCNN(num_classes=5)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 5)
    assert count_trainable_parameters(model) == 102_277


def test_model_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LandUseCNN(num_classes=1)
