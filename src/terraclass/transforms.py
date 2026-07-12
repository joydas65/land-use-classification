"""Configuration-driven image transformations for baseline parity."""

from __future__ import annotations

from torchvision import transforms

from terraclass.config import PreprocessingConfig


def build_train_transform(config: PreprocessingConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            transforms.RandomRotation(config.rotation_degrees),
            transforms.ColorJitter(
                brightness=config.color_jitter,
                contrast=config.color_jitter,
                saturation=config.color_jitter,
            ),
            transforms.ToTensor(),
            transforms.Normalize(config.normalization_mean, config.normalization_std),
        ]
    )


def build_eval_transform(config: PreprocessingConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.normalization_mean, config.normalization_std),
        ]
    )
