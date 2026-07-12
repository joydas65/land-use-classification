import pytest
import torch

from terraclass.transfer import (
    SUPPORTED_ARCHITECTURES,
    build_transfer_model,
    set_backbone_trainable,
    trainable_parameter_count,
)


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_transfer_model_output_and_freezing(architecture: str) -> None:
    model = build_transfer_model(architecture, 5, pretrained=False, dropout=0.2)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 5)
    total = trainable_parameter_count(model)
    set_backbone_trainable(model, architecture, False)
    head_only = trainable_parameter_count(model)
    assert 0 < head_only < total
    set_backbone_trainable(model, architecture, True)
    assert trainable_parameter_count(model) == total


def test_transfer_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_transfer_model("unknown", 5, pretrained=False)
