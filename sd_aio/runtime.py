"""Run provenance and detached metrics, independent of model implementations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch


class MetricWindow:
    """Mean scalar logs over one gradient accumulation window."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, values):
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Metric {name} must be scalar")
                value = value.detach()
            self.totals[name] = self.totals.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + 1

    def pop(self):
        result = {name: value / self.counts[name] for name, value in self.totals.items()}
        self.totals.clear()
        self.counts.clear()
        return result


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_metadata(output_dir, *, world_size, seed):
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "peft"):
        packages[name] = _version(name)
    files = [root / "train.py", root / "eval.py", *sorted((root / "sd_aio").rglob("*.py"))]
    payload = {
        "python": platform.python_version(),
        "packages": packages,
        "world_size": world_size,
        "seed": seed,
        "cuda": torch.version.cuda,
        "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    path = Path(output_dir) / "runtime.json"
    with (Path(output_dir) / "runtime_history.jsonl").open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def append_metrics(output_dir, step, values):
    record = {"step": int(step), **{name: float(value) for name, value in values.items()}}
    with (Path(output_dir) / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
