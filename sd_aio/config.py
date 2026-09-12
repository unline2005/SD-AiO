"""YAML configuration: shared defaults, optional bases, tasks and CLI overrides."""

from __future__ import annotations

import importlib.util
import math
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = PROJECT_ROOT / "configs" / "defaults.yaml"
_MISSING = object()
_PATH_KEYS = {
    "sd_path",
    "output_dir",
    "save_dir",
    "resume_from",
    "pretrained_encoder_path",
    "degradation_classifier_path",
    "dino_path",
    "manifest",
    "lq_path",
    "gt_path",
}
_ALLOWED_TOP_LEVEL = {
    "stage",
    "output_dir",
    "seed",
    "mixed_precision",
    "pin_memory",
    "persistent_workers",
    "keep_last_checkpoints",
    "model",
    "data",
    "optimizer",
    "scheduler",
    "loss",
    "trainer",
    "eval",
    "ema",
}


def required(cfg: OmegaConf, dotted_key: str) -> Any:
    """Read a required key. An explicitly configured null remains valid."""
    value = OmegaConf.select(cfg, dotted_key, default=_MISSING)
    if value is _MISSING:
        raise KeyError(f"Missing required configuration key: {dotted_key}")
    return value


def apply_overrides(cfg: OmegaConf, overrides: Iterable[str] | None) -> OmegaConf:
    overrides = list(overrides or [])
    for item in overrides:
        if "=" not in item or not item.partition("=")[0].strip():
            raise ValueError(f"Override must be KEY=VALUE, got {item}")
    if not overrides:
        return cfg
    merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for item in overrides:
        key, _, raw = item.partition("=")
        value = OmegaConf.to_container(OmegaConf.from_dotlist([f"value={raw}"]), resolve=False)["value"]
        OmegaConf.update(merged, key, value, merge=True)
    return merged


def _load_yaml(path: Path, ancestors: tuple[Path, ...] = ()) -> OmegaConf:
    path = path.expanduser().resolve()
    if path in ancestors:
        chain = " -> ".join(str(p) for p in (*ancestors, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = OmegaConf.load(path)
    if not OmegaConf.is_dict(config):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, (list, tuple)) and not OmegaConf.is_list(bases):
        raise TypeError(f"_base_ must be a path or list of paths: {path}")
    merged = OmegaConf.create({})
    for base in bases:
        if not isinstance(base, str) or not base.strip():
            raise ValueError(f"_base_ entries must be non-empty paths: {path}")
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = OmegaConf.merge(merged, _load_yaml(base_path, (*ancestors, path)))
    return OmegaConf.merge(merged, config)


def _resolve_single_path(value: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    # Preserve Hub IDs such as facebook/dinov2-base.
    if value.startswith(("./", "../", "~")):
        return str((PROJECT_ROOT / expanded).resolve())
    return value


def resolve_paths(cfg: OmegaConf) -> OmegaConf:
    """Resolve explicit filesystem paths against the project root in place."""

    def walk(node: Any) -> None:
        if OmegaConf.is_list(node):
            for value in node:
                walk(value)
        elif OmegaConf.is_dict(node):
            for key in list(node.keys()):
                value = node[key]
                if str(key) in _PATH_KEYS and isinstance(value, str):
                    node[key] = _resolve_single_path(value)
                else:
                    walk(value)

    walk(cfg)
    return cfg


def warn_unknown_keys(cfg: OmegaConf) -> None:
    for key in cfg:
        if str(key) not in _ALLOWED_TOP_LEVEL:
            warnings.warn(
                f"Unknown top-level config key {key}; allowed keys: {sorted(_ALLOWED_TOP_LEVEL)}",
                stacklevel=2,
            )


def _merge_tasks_file(cfg: OmegaConf) -> OmegaConf:
    tasks_file = OmegaConf.select(cfg, "data.tasks_file")
    if not tasks_file:
        return cfg
    path = Path(_resolve_single_path(str(tasks_file)))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"data.tasks_file not found: {path}")
    tasks = OmegaConf.load(path)
    if not OmegaConf.is_dict(tasks) or tasks.get("train") is None or tasks.get("test") is None:
        raise KeyError(f"Tasks file {path} must define top-level 'train' and 'test' lists")
    for split in ("train", "val", "test"):
        value = tasks.get(split, [])
        if not OmegaConf.is_list(value) and not isinstance(value, list):
            raise TypeError(f"Tasks file {path}: '{split}' must be a list")
        cfg.data[split] = value
    return cfg


def _integer(cfg: OmegaConf, key: str, minimum: int) -> None:
    value = OmegaConf.select(cfg, key, default=_MISSING)
    if value is _MISSING:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")


def validate_config(cfg: OmegaConf) -> None:
    """Check generic values without requiring training-only fields for inference."""
    stage = required(cfg, "stage")
    if not isinstance(stage, str) or not stage:
        raise KeyError("Config must define a non-empty 'stage'")
    if not stage.isidentifier() or importlib.util.find_spec(f"sd_aio.{stage}") is None:
        raise ValueError(f"Unknown stage {stage!r}; expected a stage module sd_aio/{stage}.py")
    if required(cfg, "mixed_precision") not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16 or bf16")
    if required(cfg, "eval.overall") != "task_equal":
        raise ValueError("eval.overall only supports task_equal")
    for key in (
        "trainer.train_batch_size",
        "trainer.max_steps",
        "trainer.gradient_accumulation_steps",
        "trainer.distributed_timeout_seconds",
        "keep_last_checkpoints",
        "eval.batch_size",
    ):
        _integer(cfg, key, 1)
    for key in (
        "data.num_workers",
        "trainer.log_every",
        "scheduler.warmup_steps",
        "trainer.eval_freq",
        "trainer.checkpointing_steps",
        "trainer.num_images_save_eval",
    ):
        _integer(cfg, key, 0)
    for key in ("optimizer.lr", "optimizer.head_lr", "optimizer.backbone_lr"):
        value = OmegaConf.select(cfg, key, default=_MISSING)
        if value is _MISSING:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive, got {value!r}")


def validate_training(cfg: OmegaConf) -> None:
    """Fail before allocating a model when an experiment lacks training settings."""
    cfg = OmegaConf.merge(OmegaConf.load(DEFAULTS_PATH), cfg)
    validate_config(cfg)
    for key in (
        "output_dir",
        "trainer.train_batch_size",
        "trainer.max_steps",
        "trainer.eval_freq",
        "trainer.checkpointing_steps",
        "data.train",
    ):
        if required(cfg, key) is None:
            raise ValueError(f"Training configuration key cannot be null: {key}")
    learning_rates = (
        ("optimizer.head_lr", "optimizer.backbone_lr") if cfg.stage == "classifier" else ("optimizer.lr",)
    )
    for key in learning_rates:
        if required(cfg, key) is None:
            raise ValueError(f"Training learning rate cannot be null: {key}")
    if not str(cfg.output_dir).strip():
        raise ValueError("output_dir cannot be empty")
    if not OmegaConf.is_list(cfg.data.train) or not cfg.data.train:
        raise ValueError("data.train must be a non-empty task list")
    if cfg.stage == "classifier":
        target = str(cfg.model.classifier.patch.target)
        if target not in {"none", "mean_abs", "mask_fraction", "residual"}:
            raise ValueError("model.classifier.patch.target must be none/mean_abs/mask_fraction/residual")
        weight = float(cfg.loss.patch.weight)
        if target == "none" and weight != 0:
            raise ValueError("loss.patch.weight must be zero when patch supervision is disabled")
        if target != "none" and weight <= 0:
            raise ValueError("loss.patch.weight must be positive when patch supervision is enabled")
        if target != "none" and not cfg.model.classifier.pad_to_patch_multiple:
            raise ValueError("Patch supervision requires model.classifier.pad_to_patch_multiple: true")
        threshold = float(cfg.loss.patch.mask_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("loss.patch.mask_threshold must be in [0, 1]")


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> OmegaConf:
    overrides = list(overrides or [])
    cfg = OmegaConf.merge(OmegaConf.load(DEFAULTS_PATH), _load_yaml(Path(config_path)))
    cfg = apply_overrides(cfg, overrides)
    _merge_tasks_file(cfg)
    # Explicit CLI task overrides take precedence over the shared tasks file.
    cfg = apply_overrides(cfg, overrides)
    resolve_paths(cfg)
    warn_unknown_keys(cfg)
    validate_config(cfg)
    return cfg


def resume_section(cfg: OmegaConf, key: str) -> Any:
    """Compare resolved experiment content rather than an external task-file pointer."""
    defaults = OmegaConf.load(DEFAULTS_PATH)
    value = OmegaConf.to_container(OmegaConf.merge(defaults.get(key, {}), required(cfg, key)), resolve=True)
    if key == "data" and isinstance(value, dict):
        value.pop("tasks_file", None)
    return value


def snapshot(cfg: OmegaConf, output_dir: str | Path) -> Path:
    """Atomically save a self-contained, resolved configuration for every stage."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "data" in saved:
        saved.data.pop("tasks_file", None)
    temporary = path.with_suffix(".yaml.tmp")
    OmegaConf.save(saved, temporary)
    temporary.replace(path)
    return path
