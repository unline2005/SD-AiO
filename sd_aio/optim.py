"""Native PyTorch optimizers; parameter groups remain owned by each stage."""

from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from sd_aio.config import DEFAULTS_PATH


def build_optimizer(parameters, options):
    cfg = OmegaConf.merge(OmegaConf.load(DEFAULTS_PATH).optimizer, options)
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Optimizer requires trainable parameters")
    lr = float(cfg.get("lr", cfg.get("head_lr", 0.0)))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("optimizer.lr (or head_lr for classifier groups) must be positive")
    common = {"lr": lr, "weight_decay": float(cfg.weight_decay)}
    if not math.isfinite(common["weight_decay"]) or common["weight_decay"] < 0:
        raise ValueError("optimizer.weight_decay must be finite and nonnegative")
    name = str(cfg.name).lower()
    if name in {"adam", "adamw"}:
        optimizer = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return optimizer(parameters, betas=tuple(cfg.betas), eps=float(cfg.eps), **common)
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=float(cfg.momentum), **common)
    raise ValueError(f"Unknown optimizer: {cfg.name}; choose adamw, adam, or sgd")
