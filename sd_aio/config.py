"""Configuration loading.

Every experiment is a YAML file merged on top of ``configs/defaults.yaml``.
There are intentionally almost no implicit defaults in code: values shared by
all experiments live in the visible ``defaults.yaml``; stage-specific values
live in the stage YAML; everything missing is an error.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = PROJECT_ROOT / "configs" / "defaults.yaml"

_MISSING = object()

# Keys that should be converted to absolute paths (relative to the project root).
_PATH_KEYS = {
    "sd_path",
    "output_dir",
    "save_dir",
    "resume_from",
    "pretrained_encoder_path",
    "degradation_classifier_path",
    "dino_path",
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
    """Return ``cfg.a.b.c`` or raise if the key does not exist.

    A key that exists with value ``None`` is returned as ``None``; only a truly
    missing key is an error.
    """
    value = OmegaConf.select(cfg, dotted_key, default=_MISSING)
    if value is _MISSING:
        raise KeyError(f"Missing required configuration key: {dotted_key}")
    return value


def get(cfg: OmegaConf, dotted_key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, dotted_key, default=_MISSING)
    return default if value is _MISSING else value


def apply_overrides(cfg: OmegaConf, overrides: Iterable[str] | None) -> OmegaConf:
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item}")
    merged = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, merged)


def _resolve_single_path(value: str) -> str:
    value = str(value)
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    # Only filesystem-looking relative paths are resolved. Bare ids such as
    # "facebook/dinov2-base" must keep working as Hugging Face repo ids.
    if value.startswith(("./", "../", "~")):
        return str((PROJECT_ROOT / expanded).resolve())
    return value


def resolve_paths(cfg: OmegaConf) -> OmegaConf:
    """Turn project-relative filesystem paths into absolute paths in-place."""

    def walk(node: Any) -> Any:
        if OmegaConf.is_list(node):
            for item in node:
                walk(item)
            return node
        if OmegaConf.is_dict(node):
            for key in list(node.keys()):
                value = node[key]
                if str(key) in _PATH_KEYS and isinstance(value, str):
                    node[key] = _resolve_single_path(value)
                else:
                    walk(value)
            return node
        return node

    walk(cfg)
    return cfg


def warn_unknown_keys(cfg: OmegaConf) -> None:
    """Warn about top-level typos (the old ``--lora_rank_vae`` class of drift)."""
    for key in cfg:
        if str(key) not in _ALLOWED_TOP_LEVEL:
            warnings.warn(
                f"Unknown top-level config key {key}; allowed keys: {sorted(_ALLOWED_TOP_LEVEL)}",
                stacklevel=2,
            )


def _merge_tasks_file(cfg: OmegaConf) -> OmegaConf:
    """Merge the shared ``data.tasks_file`` dataset definition into ``data.train/test``."""
    if not OmegaConf.is_dict(cfg.get("data")):
        return cfg
    tasks_file = cfg.data.get("tasks_file")
    if not tasks_file:
        return cfg
    tasks_file = Path(_resolve_single_path(str(tasks_file)))
    if not tasks_file.is_absolute():
        tasks_file = (PROJECT_ROOT / tasks_file).resolve()
    if not tasks_file.exists():
        raise FileNotFoundError(f"data.tasks_file not found: {tasks_file}")
    tasks = OmegaConf.load(tasks_file)
    if tasks.get("train") is None or tasks.get("test") is None:
        raise KeyError(f"Tasks file {tasks_file} must define top-level 'train' and 'test' lists")
    cfg.data.train = tasks.train
    cfg.data.test = tasks.test
    return cfg


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> OmegaConf:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base = OmegaConf.load(DEFAULTS_PATH)
    experiment = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base, experiment)
    cfg = apply_overrides(cfg, overrides)
    _merge_tasks_file(cfg)
    resolve_paths(cfg)
    warn_unknown_keys(cfg)

    if not required(cfg, "stage"):
        raise KeyError("Config must define a non-empty 'stage'")
    return cfg


def snapshot(cfg: OmegaConf, output_dir: str | Path) -> Path:
    """Save the fully-resolved config next to a run's checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    OmegaConf.save(cfg, path)
    return path
