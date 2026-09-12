"""Pixel objectives shared by restoration stages. Inputs use [-1, 1]."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pixel_loss(prediction, target, kind: str, *, epsilon: float):
    if prediction.shape != target.shape:
        raise ValueError(f"Pixel loss shape mismatch: {prediction.shape} != {target.shape}")
    prediction, target = prediction.float(), target.float()
    if kind == "l1":
        return F.l1_loss(prediction, target)
    if kind == "mse":
        return F.mse_loss(prediction, target)
    if kind == "charbonnier":
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Charbonnier epsilon must be finite and positive")
        return torch.sqrt((prediction - target).square() + epsilon**2).mean()
    raise ValueError(f"Unknown pixel loss: {kind}; choose l1, mse, or charbonnier")
