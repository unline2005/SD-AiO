"""Trainable weights, resumable checkpoints and exponential moving averages."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

WEIGHTS_NAME = "weights.safetensors"
EMA_NAME = "ema.safetensors"
OPTIMIZER_NAME = "optimizer.pt"
CHECKPOINT_PREFIX = "checkpoint-"
COMPLETE_NAME = "complete.json"
WRITING_NAME = ".incomplete"


def save_model_weights(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("Refusing to save a model without trainable parameters")
    if hasattr(model, "save_auxiliary"):
        model.save_auxiliary(path)
    temporary = path.with_suffix(".tmp")
    save_file(state, temporary)
    temporary.replace(path)
    return path


def load_model_weights(model: nn.Module, path: str | Path) -> None:
    """Validate every supplied tensor before loading; absent frozen state is expected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = load_file(str(path), device="cpu")
    if not state:
        raise RuntimeError(f"Checkpoint {path} is empty")
    expected = model.state_dict()
    unexpected = set(state) - set(expected)
    if unexpected:
        raise RuntimeError(
            f"Checkpoint {path} has {len(unexpected)} unexpected keys; "
            f"config/checkpoint mismatch. First: {sorted(unexpected)[:5]}"
        )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = trainable - set(state)
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} is missing {len(missing)} trainable keys; first: {sorted(missing)[:5]}"
        )
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise RuntimeError(
                f"Checkpoint {path}: shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(expected[name].shape)}"
            )
    # Only absent frozen keys are permitted; trainable keys and shapes were checked above.
    model.load_state_dict(state, strict=False)
    if hasattr(model, "load_auxiliary"):
        auxiliary_source = path
        # Legacy EMA files shared the ordinary weights' frozen condition sidecar.
        if (
            path.name == EMA_NAME
            and not path.with_name("ema.condition.safetensors").exists()
            and path.with_name("weights.condition.safetensors").is_file()
        ):
            auxiliary_source = path.with_name(WEIGHTS_NAME)
        model.load_auxiliary(auxiliary_source)


def _step_from_dir(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def is_checkpoint_complete(path: str | Path) -> bool:
    """Accept legacy complete directories; new checkpoints must match their manifest."""
    path = Path(path)
    try:
        if not path.is_dir() or path.is_symlink() or _step_from_dir(path) < 0:
            return False
        if (path / WRITING_NAME).exists() or any(path.glob("*.tmp")):
            return False
        required = {WEIGHTS_NAME, OPTIMIZER_NAME}
        if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
            return False
        manifest = path / COMPLETE_NAME
        if not manifest.exists():
            return True
        content = json.loads(manifest.read_text())
        files = content["files"]
        if content["version"] != 1 or content["step"] != _step_from_dir(path):
            return False
        if not isinstance(files, dict) or not required.issubset(files):
            return False
        return all(
            Path(name).name == name
            and isinstance(size, int)
            and size > 0
            and (path / name).is_file()
            and (path / name).stat().st_size == size
            for name, size in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def iter_checkpoints(output_dir: str | Path) -> list[Path]:
    directory = Path(output_dir) / "checkpoints"
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if is_checkpoint_complete(path)),
        key=_step_from_dir,
        reverse=True,
    )


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    candidates = iter_checkpoints(output_dir)
    return candidates[0] if candidates else None


def resolve_weights_path(path_or_dir: str | Path, prefer_ema: bool = False) -> Path:
    """Resolve a weight file, checkpoint directory, final directory or experiment root."""
    path = Path(path_or_dir)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got {path}")
        if prefer_ema and path.name == WEIGHTS_NAME:
            raise ValueError("--use_ema was set but an explicit weights.safetensors file was provided")
        parent = path.parent
        if parent.name.startswith(CHECKPOINT_PREFIX) and (
            (parent / WRITING_NAME).exists()
            or ((parent / COMPLETE_NAME).exists() and not is_checkpoint_complete(parent))
        ):
            raise RuntimeError(f"Checkpoint is incomplete: {parent}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    name = EMA_NAME if prefer_ema else WEIGHTS_NAME
    if path.name.startswith(CHECKPOINT_PREFIX):
        if (path / WRITING_NAME).exists() or (
            (path / COMPLETE_NAME).exists() and not is_checkpoint_complete(path)
        ):
            raise RuntimeError(f"Checkpoint is incomplete: {path}")
        directory = path
    elif (path / name).is_file():
        directory = path
    else:
        latest = find_latest_checkpoint(path)
        directory = latest if latest is not None else path / "final"
    weights = directory / name
    if not weights.is_file():
        prefix = "--use_ema was set but " if prefer_ema else ""
        raise FileNotFoundError(f"{prefix}{weights} does not exist")
    return weights


def _save_ema_weights(model: nn.Module, ema: ModelEMA, path: Path) -> None:
    if hasattr(model, "save_auxiliary"):
        model.save_auxiliary(path)
    temporary = path.with_suffix(".tmp")
    save_file(ema.state_dict(), temporary)
    temporary.replace(path)


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
    step = int(step)
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    directory = Path(output_dir) / "checkpoints" / f"checkpoint-{step:08d}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Unfinished staging directories are invisible to discovery and retention.
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        staged = Path(temporary)
        save_model_weights(model, staged / WEIGHTS_NAME)
        if ema is not None:
            _save_ema_weights(model, ema, staged / EMA_NAME)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step": step,
            },
            staged / OPTIMIZER_NAME,
        )
        files = {path.name: path.stat().st_size for path in staged.iterdir() if path.is_file()}
        (staged / COMPLETE_NAME).write_text(
            json.dumps({"version": 1, "step": step, "files": files}, indent=2)
        )
        _publish_checkpoint(staged, directory)
    for stale in iter_checkpoints(output_dir)[keep_last:]:
        shutil.rmtree(stale)
    return directory


def _publish_checkpoint(staged: Path, directory: Path) -> None:
    if not directory.exists():
        staged.replace(directory)
        return
    if not is_checkpoint_complete(directory):
        raise RuntimeError(f"Refusing to overwrite an incomplete or active checkpoint: {directory}")
    # A repeated save of the same step keeps the old complete directory until publication.
    backup = Path(tempfile.mkdtemp(prefix=f".{directory.name}-old-", dir=directory.parent))
    directory.replace(backup)
    try:
        staged.replace(directory)
    except BaseException:
        backup.replace(directory)
        raise
    shutil.rmtree(backup)


def _checkpoint_dirs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        raise ValueError(
            "resume_from must be an output_dir or checkpoint-* directory; "
            "a weights-only file cannot restore optimizer state"
        )
    return [path] if path.name.startswith(CHECKPOINT_PREFIX) else iter_checkpoints(path)


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
    for directory in candidates:
        weights = directory / WEIGHTS_NAME
        optimizer_file = directory / OPTIMIZER_NAME
        try:
            if not weights.is_file():
                raise FileNotFoundError(f"{weights} does not exist")
            if not optimizer_file.is_file():
                raise FileNotFoundError(f"{optimizer_file} does not exist")
            if not is_checkpoint_complete(directory):
                raise RuntimeError(f"Checkpoint is incomplete: {directory}")
            state = torch.load(optimizer_file, map_location="cpu", weights_only=True)
            step = int(state["step"])
            if step != _step_from_dir(directory):
                raise ValueError(
                    f"Optimizer step {step} disagrees with checkpoint directory {directory.name}"
                )
            if scheduler is not None and state["scheduler"] is None:
                raise ValueError(f"Checkpoint {directory} has no scheduler state")
            load_model_weights(model, weights)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            return step, directory
        except Exception as exc:
            last_error = exc
            warnings.warn(
                f"Checkpoint {directory} failed to load ({exc}); trying previous checkpoint",
                stacklevel=2,
            )
    raise RuntimeError(f"All checkpoints failed to load: {last_error}")


def save_final(model: nn.Module, output_dir: str | Path, ema: ModelEMA | None = None) -> Path:
    directory = Path(output_dir) / "final"
    directory.mkdir(parents=True, exist_ok=True)
    save_model_weights(model, directory / WEIGHTS_NAME)
    if ema is not None:
        _save_ema_weights(model, ema, directory / EMA_NAME)
    else:
        (directory / EMA_NAME).unlink(missing_ok=True)
    return directory


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
        for name, tensor in state.items():
            self.shadow[name].copy_(tensor.to(self.shadow[name].device))
