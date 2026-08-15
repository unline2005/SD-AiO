"""Small shared helpers that do not belong to any single stage."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def set_train_mode(module: nn.Module) -> None:
    """Put every *trainable* leaf module in train mode and keep frozen ones in eval mode.

    This is the exact opposite of the old ``model.train()`` bug: frozen
    GroupNorm/running-statistics modules never get switched back to batch
    statistics during training.
    """

    def _set(module: nn.Module) -> None:
        children = list(module.children())
        if not children:
            module.train()
            return
        for child in children:
            if any(p.requires_grad for p in child.parameters()):
                _set(child)
            else:
                child.eval()

    _set(module)


def set_eval_mode(module: nn.Module) -> None:
    """Set the whole model and every child to eval mode."""
    module.eval()


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Move tensor fields to ``device`` and cast floating-point fields to ``dtype``.

    Non-tensor fields (prompts, task names, ...) are left untouched.
    """
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device=device)
            if value.dtype in FLOAT_DTYPES:
                value = value.to(dtype=dtype)
        out[key] = value
    return out


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def weight_dtype_for(mixed_precision: str | None) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision or "no", torch.float32)
