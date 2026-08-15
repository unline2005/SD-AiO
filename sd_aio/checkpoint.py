"""Checkpoint helpers: safetensors weights + torch optimizer state + EMA."""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

WEIGHTS_NAME = "weights.safetensors"
EMA_NAME = "ema.safetensors"
OPTIMIZER_NAME = "optimizer.pt"
CHECKPOINT_PREFIX = "checkpoint-"


def save_model_weights(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("Refusing to save a model without trainable parameters")
    save_file(state, path)
    return path


def load_model_weights(model: nn.Module, path: str | Path) -> None:
    """Load trainable-only weights.

    Unexpected keys and missing *trainable* keys are fatal.  Missing frozen
    keys are expected because frozen SD/DINO weights are never stored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = load_file(str(path), device="cpu")
    if not state:
        raise RuntimeError(f"Checkpoint {path} is empty")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Checkpoint {path} has {len(unexpected)} unexpected keys; "
            f"config/checkpoint mismatch. First: {unexpected[:5]}"
        )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = trainable & set(missing)
    if missing_trainable:
        raise RuntimeError(
            f"Checkpoint {path} is missing {len(missing_trainable)} trainable keys; "
            f"first: {sorted(missing_trainable)[:5]}"
        )


def _step_from_dir(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def iter_checkpoints(output_dir: str | Path) -> list[Path]:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX)]
    return sorted(candidates, key=_step_from_dir, reverse=True)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    candidates = iter_checkpoints(output_dir)
    return candidates[0] if candidates else None


def resolve_weights_path(path_or_dir: str | Path, prefer_ema: bool = False) -> Path:
    """Resolve a user path to a concrete ``weights.safetensors``/``ema.safetensors`` file."""
    path = Path(path_or_dir)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got {path}")
        if prefer_ema and path.name == WEIGHTS_NAME:
            raise ValueError("--use_ema was set but an explicit weights.safetensors file was provided")
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.name.startswith(CHECKPOINT_PREFIX):
        checkpoint_dir = path
    else:
        checkpoints = iter_checkpoints(path)
        if checkpoints:
            checkpoint_dir = checkpoints[0]
        else:
            final_weights = path / "final" / (EMA_NAME if prefer_ema else WEIGHTS_NAME)
            if final_weights.exists():
                return final_weights
            if prefer_ema:
                raise FileNotFoundError(f"--use_ema was set but {final_weights} does not exist")
            raise FileNotFoundError(f"No checkpoints or final weights found under {path}")

    if prefer_ema:
        ema_weights = checkpoint_dir / EMA_NAME
        if not ema_weights.exists():
            raise FileNotFoundError(f"--use_ema was set but {ema_weights} does not exist")
        return ema_weights
    weights = checkpoint_dir / WEIGHTS_NAME
    if not weights.exists():
        raise FileNotFoundError(f"No {weights.name} in {checkpoint_dir}")
    return weights


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    step: int,
    output_dir: str | Path,
    ema: ModelEMA | None = None,
    keep_last: int = 3,
) -> Path:
    keep_last = int(keep_last)
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"checkpoint-{int(step):08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_model_weights(model, checkpoint_dir / WEIGHTS_NAME)
    if ema is not None:
        save_file(ema.state_dict(), checkpoint_dir / EMA_NAME)
    else:
        (checkpoint_dir / EMA_NAME).unlink(missing_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": int(step),
        },
        checkpoint_dir / OPTIMIZER_NAME,
    )

    for stale in iter_checkpoints(output_dir)[keep_last:]:
        shutil.rmtree(stale, ignore_errors=True)
    return checkpoint_dir


def _checkpoint_dirs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        raise ValueError(
            "resume_from must be an output_dir or checkpoint-* directory; "
            "a weights-only file cannot restore optimizer state"
        )
    if path.name.startswith(CHECKPOINT_PREFIX):
        return [path]
    return iter_checkpoints(path)


def restore_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    path: str | Path,
) -> tuple[int, Path]:
    """Resume from the newest usable checkpoint; corrupt checkpoints are skipped."""
    candidates = _checkpoint_dirs(path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {path}")

    last_error: Exception | None = None
    for checkpoint_dir in candidates:
        weights = checkpoint_dir / WEIGHTS_NAME
        optimizer_file = checkpoint_dir / OPTIMIZER_NAME
        try:
            if not weights.is_file():
                raise FileNotFoundError(f"{weights} does not exist")
            if not optimizer_file.is_file():
                raise FileNotFoundError(f"{optimizer_file} does not exist")
            load_model_weights(model, weights)

            state = torch.load(optimizer_file, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            return int(state["step"]), checkpoint_dir
        except Exception as exc:
            last_error = exc
            warnings.warn(
                f"Checkpoint {checkpoint_dir} failed to load ({exc}); trying previous checkpoint",
                stacklevel=2,
            )

    raise RuntimeError(f"All checkpoints failed to load: {last_error}")


def save_final(model: nn.Module, output_dir: str | Path, ema: ModelEMA | None = None) -> Path:
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_model_weights(model, final_dir / WEIGHTS_NAME)
    if ema is not None:
        save_file(ema.state_dict(), final_dir / EMA_NAME)
    else:
        (final_dir / EMA_NAME).unlink(missing_ok=True)
    return final_dir


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, path: str | Path) -> None:
        state = load_file(str(path), device="cpu")
        missing = set(self.shadow) - set(state)
        unexpected = set(state) - set(self.shadow)
        if missing or unexpected:
            raise RuntimeError(
                f"EMA checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
            )
        for name, tensor in state.items():
            if tensor.shape != self.shadow[name].shape:
                raise ValueError(
                    f"EMA parameter {name} shape mismatch: "
                    f"{tuple(tensor.shape)} != {tuple(self.shadow[name].shape)}"
                )
            self.shadow[name].copy_(tensor.to(self.shadow[name].device))
