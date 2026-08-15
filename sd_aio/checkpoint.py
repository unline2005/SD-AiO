"""Checkpoint and resume helpers.

Layout (all under ``output_dir``)::

    output_dir/
    ├── config.yaml                     # fully resolved build snapshot
    ├── checkpoints/
    │   └── checkpoint-00001000/
    │       ├── weights.safetensors     # trainable parameters only
    │       ├── ema.safetensors         # optional EMA shadow
    │       ├── optimizer.pt            # optimizer + scheduler + step
    │       └── state.json
    └── final/
        ├── weights.safetensors
        └── ema.safetensors             # optional

Model weights are always ``safetensors``; optimizer/scheduler state is plain
``torch.save`` because those objects cannot be serialised safely.
"""

from __future__ import annotations

import json
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
STATE_NAME = "state.json"
CHECKPOINT_PREFIX = "checkpoint-"


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only ``requires_grad=True`` parameters. Frozen SD/DINO weights are never duplicated."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_model_weights(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    if not state:
        raise RuntimeError("Refusing to save an empty model: no trainable parameters found")
    save_file(state, path)
    return path


def load_model_weights(model: nn.Module, path: str | Path) -> tuple[list[str], list[str]]:
    """Load a safetensors checkpoint and fail on real mismatches.

    ``missing`` keys are allowed only when they belong to frozen parameters
    (they are intentionally not stored).  ``unexpected`` keys are always fatal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    if unexpected:
        raise RuntimeError(
            f"Checkpoint {path} has {len(unexpected)} unexpected keys "
            f"(config/checkpoint mismatch); first: {unexpected[:5]}"
        )
    named_parameters = dict(model.named_parameters())
    trainable_missing = [
        key
        for key in missing
        if named_parameters.get(key) is not None and named_parameters[key].requires_grad
    ]
    if trainable_missing:
        raise RuntimeError(
            f"Checkpoint {path} is missing {len(trainable_missing)} trainable keys; "
            f"first: {trainable_missing[:5]}"
        )
    return missing, unexpected


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
    """Resolve a user-provided path to a concrete safetensors file."""
    path = Path(path_or_dir)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got {path}")
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
            final_dir = path / "final"
            weights = final_dir / WEIGHTS_NAME
            if weights.exists():
                return weights
            raise FileNotFoundError(f"No checkpoints or final weights found under {path}")

    if prefer_ema and (checkpoint_dir / EMA_NAME).exists():
        return checkpoint_dir / EMA_NAME
    return checkpoint_dir / WEIGHTS_NAME


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    step: int,
    output_dir: str | Path,
    ema: ModelEMA | None = None,
    keep_last: int = 3,
) -> Path:
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"checkpoint-{int(step):08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_model_weights(model, checkpoint_dir / WEIGHTS_NAME)
    if ema is not None:
        save_file(ema.state_dict(), checkpoint_dir / EMA_NAME)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None,
            "step": int(step),
        },
        checkpoint_dir / OPTIMIZER_NAME,
    )
    (checkpoint_dir / STATE_NAME).write_text(json.dumps({"step": int(step)}, indent=2))

    for stale in iter_checkpoints(output_dir)[max(0, int(keep_last)) :]:
        shutil.rmtree(stale, ignore_errors=True)
    return checkpoint_dir


def restore_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    path: str | Path,
    prefer_ema: bool = False,
) -> tuple[int, Path]:
    """Restore weights + optimizer + scheduler, falling back to the previous checkpoint.

    Returns ``(step, checkpoint_dir)``.  A corrupt latest checkpoint is skipped
    with a warning (VOSR-style rollback) instead of killing a long run.
    """
    path = Path(path)
    if path.is_file():
        candidates = [path.parent if path.name in (WEIGHTS_NAME, EMA_NAME) else path]
    elif path.name.startswith(CHECKPOINT_PREFIX):
        candidates = [path]
    else:
        candidates = iter_checkpoints(path)

    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {path}")

    last_error: Exception | None = None
    for checkpoint_dir in candidates:
        weights = checkpoint_dir / (EMA_NAME if prefer_ema else WEIGHTS_NAME)
        if not weights.exists() and prefer_ema:
            weights = checkpoint_dir / WEIGHTS_NAME
        if not weights.exists():
            last_error = FileNotFoundError(f"No {WEIGHTS_NAME} in {checkpoint_dir}")
            warnings.warn(f"Skipping checkpoint {checkpoint_dir}: {last_error}", stacklevel=2)
            continue
        try:
            load_model_weights(model, weights)
            step = 0
            state_file = checkpoint_dir / STATE_NAME
            if state_file.exists():
                step = int(json.loads(state_file.read_text()).get("step", 0))

            optimizer_file = checkpoint_dir / OPTIMIZER_NAME
            if optimizer_file.exists():
                training_state = torch.load(optimizer_file, map_location="cpu", weights_only=False)
                optimizer.load_state_dict(training_state["optimizer"])
                if scheduler is not None and training_state.get("scheduler") is not None:
                    scheduler.load_state_dict(training_state["scheduler"])
                step = int(training_state.get("step", step))
            elif step == 0:
                last_error = FileNotFoundError(f"No {OPTIMIZER_NAME} in {checkpoint_dir}")
                warnings.warn(f"Skipping checkpoint {checkpoint_dir}: {last_error}", stacklevel=2)
                continue
            return step, checkpoint_dir
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
    return final_dir


class ModelEMA:
    """Exponential moving average over trainable parameters only."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow and parameter.requires_grad:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, path: str | Path) -> None:
        state = load_file(str(path), device="cpu")
        for name, tensor in state.items():
            if name not in self.shadow:
                raise KeyError(f"EMA checkpoint contains unexpected parameter {name}")
            if tensor.shape != self.shadow[name].shape:
                raise ValueError(
                    f"EMA parameter {name} shape mismatch: {tuple(tensor.shape)} != {tuple(self.shadow[name].shape)}"
                )
            self.shadow[name].copy_(tensor.to(self.shadow[name].device))
