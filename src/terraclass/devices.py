"""Shared device selection without importing the training-only dependency stack."""

from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic Torch device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of auto, cpu, cuda, or mps")
    return torch.device(requested)
