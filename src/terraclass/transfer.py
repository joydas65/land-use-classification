"""Supported ImageNet transfer-learning architectures and freezing policy."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

SUPPORTED_ARCHITECTURES = ("resnet18", "efficientnet_b0")


def build_transfer_model(
    architecture: str,
    class_count: int,
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a supported classifier with a task-specific output head."""
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, class_count))
        return model
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, class_count)
        )
        return model
    raise ValueError(
        f"Unsupported architecture {architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}"
    )


def set_backbone_trainable(model: nn.Module, architecture: str, trainable: bool) -> None:
    """Freeze or unfreeze the feature extractor while always training the classifier head."""
    if architecture == "resnet18":
        classifier = model.fc
    elif architecture == "efficientnet_b0":
        classifier = model.classifier
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
