"""Disk datasets, synchronized geometry and task-balanced data loaders.

Images use RGB tensors in [-1, 1]. Online Gaussian noise is seeded by task
and sample index during evaluation; stored noisy/clean pairs are kept intact.
"""

from __future__ import annotations

import json
import math
import random
import re
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision.transforms import functional as TF

from sd_aio.config import DEFAULTS_PATH

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_DEFAULTS = OmegaConf.load(DEFAULTS_PATH)


def list_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted images, rejecting absent or empty directories."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    entries = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"Image directory contains no supported images: {directory}")
    return files


def read_manifest(task: dict[str, Any], *, require_gt: bool = True) -> list[dict[str, Path]]:
    """Read ordered JSON/JSONL records with explicit lq/gt paths.

    Relative paths resolve against manifest_root, or the manifest directory.
    Repeated LQ paths are rejected; several LQ images may share one GT.
    Classification may omit GT. Labels remain explicit in the task config.
    """
    path = Path(task["manifest"]).resolve()
    if task.get("lq_path") or task.get("gt_path"):
        raise ValueError("Use either manifest or lq_path/gt_path, not both")
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"Manifest must be JSON or JSONL: {path}")
    with path.open(encoding="utf-8") as source:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in source if line.strip()]
        else:
            records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a non-empty list of records: {path}")
    root = Path(task.get("manifest_root") or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    result = []
    seen = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        row = {}
        for key in ("lq", "gt") if require_gt else ("lq",):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path}: record {index} requires a non-empty {key} path")
            image = Path(value)
            image = (image if image.is_absolute() else root / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"{path}: record {index} image not found: {image}")
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"{path}: unsupported image format: {image}")
            row[key] = image
        if row["lq"] in seen:
            raise ValueError(f"{path}: duplicate LQ image: {row['lq']}")
        seen.add(row["lq"])
        result.append(row)
    return result


def _image_index(files: list[Path], key, label: str) -> dict[str, Path]:
    index = {}
    for path in files:
        identity = key(path)
        if identity in index:
            raise ValueError(f"Ambiguous {label} {identity!r}: {index[identity]} and {path}")
        index[identity] = path
    return index


def pair_images(task: dict[str, Any]) -> list[tuple[Path, Path]]:
    """Pair directories by stem/prefix/relative_stem, or use an explicit manifest."""
    if task.get("manifest"):
        return [(row["lq"], row["gt"]) for row in read_manifest(task)]
    lq_root, gt_root = Path(task["lq_path"]), Path(task["gt_path"])
    lq_sub, gt_sub = task.get("lq_subdir"), task.get("gt_subdir")
    if bool(lq_sub) != bool(gt_sub):
        raise ValueError("lq_subdir and gt_subdir must be provided together")
    lq_dir = lq_root / lq_sub if lq_sub else lq_root
    gt_dir = gt_root / gt_sub if gt_sub else gt_root
    recursive = bool(task.get("recursive", False))
    lq_files, gt_files = list_images(lq_dir, recursive), list_images(gt_dir, recursive)
    prefix_len = int(task.get("match_prefix_len", 0) or 0)
    pairing = str(task.get("pairing", "prefix" if prefix_len else "stem"))
    if pairing not in {"stem", "prefix", "relative_stem"}:
        raise ValueError(f"Unsupported pairing strategy: {pairing}")
    if prefix_len < 0 or (pairing == "prefix" and prefix_len == 0):
        raise ValueError("Prefix matching requires positive match_prefix_len")
    if prefix_len and pairing != "prefix":
        raise ValueError("match_prefix_len requires pairing: prefix")

    def identity(path, root):
        if pairing == "relative_stem":
            return str(path.relative_to(root).with_suffix(""))
        return path.stem[:prefix_len] if pairing == "prefix" else path.stem

    gt_index = _image_index(gt_files, lambda path: identity(path, gt_dir), "GT pairing key")
    if pairing != "prefix":
        _image_index(lq_files, lambda path: identity(path, lq_dir), "LQ pairing key")
        if len(lq_files) != len(gt_files):
            raise ValueError(f"[{task.get('name', lq_root)}] LQ ({len(lq_files)}) != GT ({len(gt_files)})")
    pairs = []
    for lq in lq_files:
        gt = gt_index.get(identity(lq, lq_dir))
        if gt is None:
            raise ValueError(f"[{task.get('name', lq_root)}] no GT {pairing} match for {lq}")
        pairs.append((lq, gt))
    return pairs


def parse_sigma_from_name(task_name: str) -> float | None:
    """Parse ``..._15`` / ``..._25.5`` suffixes used by denoise task names."""
    match = re.search(r"_(\d+(?:\.\d+)?)$", task_name)
    return float(match.group(1)) if match else None


def add_gaussian_noise(image: torch.Tensor, sigma: float, seed: int | None = None) -> torch.Tensor:
    """Add Gaussian noise in [-1, 1] space; ``sigma`` is in [0, 255] pixel space."""
    if sigma <= 0:
        return image
    if seed is not None:
        generator = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        noise = noise * (sigma / 127.5)
    else:
        noise = torch.randn_like(image) * (sigma / 127.5)
    return (image + noise).clamp(-1.0, 1.0)


def build_deg_types(task_entries: Sequence[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(task["deg_type"]) for task in task_entries))


def task_sampling_weight(task):
    weight = float(task.get("sampling_weight", 1.0))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("sampling_weight must be positive and finite")
    return weight


class PairedTransform:
    """Shared geometry and [-1, 1] RGB conversion for aligned image pairs.

    The legacy defaults remain random square crops for training and native
    resolution for evaluation. Crops upscale small inputs to avoid black fill.
    """

    MODES: ClassVar[set[str]] = {"native", "random_crop", "center_crop", "resize_short_center_crop", "resize"}
    INTERPOLATIONS: ClassVar[dict[str, TF.InterpolationMode]] = {
        "nearest": TF.InterpolationMode.NEAREST,
        "bilinear": TF.InterpolationMode.BILINEAR,
        "bicubic": TF.InterpolationMode.BICUBIC,
        "lanczos": TF.InterpolationMode.LANCZOS,
    }

    def __init__(
        self,
        image_size: int,
        is_train: bool = True,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rot90_prob: float = 0.0,
        *,
        mode: str | None = None,
        interpolation: str = "bicubic",
        exif_transpose: bool = False,
    ) -> None:
        self.image_size = image_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.mode = mode if mode is not None else ("random_crop" if is_train else "native")
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown preprocessing mode: {self.mode}")
        if self.mode != "native" and image_size <= 0:
            raise ValueError("Cropped/resized image_size must be positive")
        if not is_train and self.mode == "random_crop":
            raise ValueError("Evaluation preprocessing cannot use random_crop")
        if interpolation not in self.INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation: {interpolation}")
        if any(not 0 <= prob <= 1 for prob in (hflip_prob, vflip_prob, rot90_prob)):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        self.interpolation = self.INTERPOLATIONS[interpolation]
        self.exif_transpose = exif_transpose

    def geometry(self, lq_image: Image.Image, gt_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.exif_transpose:
            lq_image, gt_image = ImageOps.exif_transpose(lq_image), ImageOps.exif_transpose(gt_image)
        if lq_image.size != gt_image.size:
            raise ValueError(
                f"Paired images must have identical sizes: LQ={lq_image.size}, GT={gt_image.size}"
            )
        if self.mode == "native":
            return lq_image, gt_image
        w, h = gt_image.size
        target = self.image_size
        if self.mode == "resize":
            size = (target, target)
            return tuple(TF.resize(image, size, self.interpolation) for image in (lq_image, gt_image))
        if self.mode == "resize_short_center_crop" or min(h, w) < target:
            scale = target / min(h, w)
            h, w = round(h * scale), round(w * scale)
            lq_image = TF.resize(lq_image, (h, w), self.interpolation)
            gt_image = TF.resize(gt_image, (h, w), self.interpolation)
        if self.mode == "random_crop":
            top, left = random.randint(0, h - target), random.randint(0, w - target)
        else:
            top, left = (h - target) // 2, (w - target) // 2
        return TF.crop(lq_image, top, left, target, target), TF.crop(gt_image, top, left, target, target)

    def __call__(self, lq_image: Image.Image, gt_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        lq_image, gt_image = self.geometry(lq_image, gt_image)
        if self.is_train:
            if random.random() < self.hflip_prob:
                lq_image, gt_image = TF.hflip(lq_image), TF.hflip(gt_image)
            if random.random() < self.vflip_prob:
                lq_image, gt_image = TF.vflip(lq_image), TF.vflip(gt_image)
            if random.random() < self.rot90_prob:
                angle = random.choice([90, -90])
                lq_image, gt_image = TF.rotate(lq_image, angle), TF.rotate(gt_image, angle)
        return TF.to_tensor(lq_image) * 2.0 - 1.0, TF.to_tensor(gt_image) * 2.0 - 1.0


def preprocessing_options(cfg: OmegaConf, kind: str, task: dict[str, Any]) -> dict[str, Any]:
    """Merge shared YAML geometry with an explicit per-task override.

    An absent node preserves the constructor API for older saved configs.
    Public defaults for all four kinds live in configs/defaults.yaml.
    """
    node = OmegaConf.select(cfg, f"data.preprocessing.{kind}")
    options = dict(OmegaConf.to_container(node, resolve=True)) if node is not None else {}
    override = task.get("preprocessing", {})
    if not isinstance(override, dict):
        raise ValueError("Task preprocessing must be a mapping")
    options.update(override)
    unknown = set(options) - {"mode", "image_size", "interpolation", "exif_transpose"}
    if unknown:
        raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
    return options


class PairedImageDataset(Dataset):
    """One task: paired LQ/GT images with optional online Gaussian degradation."""

    def __init__(self, task: dict[str, Any], transform: PairedTransform) -> None:
        self.task_name = str(task["name"])
        self.deg_type = str(task["deg_type"])
        self.transform = transform
        self.repeat_ratio = max(1, int(task.get("repeat_ratio", 1) or 1))

        self.pairs = pair_images(task)
        self.is_denoise = task.get("noise_sigma") is not None or (
            task.get("deg_type") == "noise" and all(lq == gt for lq, gt in self.pairs)
        )

        explicit_sigma = task.get("noise_sigma")
        if explicit_sigma is not None:
            self.sigma = float(explicit_sigma)
        elif self.is_denoise:
            parsed = parse_sigma_from_name(self.task_name)
            if parsed is None:
                raise ValueError(f"Denoise task {self.task_name} must end with _<sigma> or set noise_sigma")
            self.sigma = parsed
        else:
            self.sigma = 0.0

        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("noise_sigma must be finite and non-negative")
        excluded = task.get("exclude_filenames", [])
        if excluded:
            if not isinstance(excluded, (list, tuple)) or len(set(excluded)) != len(excluded):
                raise ValueError("exclude_filenames must be a unique list")
            missing = set(excluded) - {lq.name for lq, _ in self.pairs}
            if missing:
                raise ValueError(f"Excluded paired filenames do not exist: {sorted(missing)}")
            self.pairs = [(lq, gt) for lq, gt in self.pairs if lq.name not in excluded]
            if not self.pairs:
                raise ValueError("No image pairs remain after exclusions")
        self.indices = list(range(len(self.pairs))) * self.repeat_ratio

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        real_index = self.indices[index]
        lq_path, gt_path = self.pairs[real_index]
        with Image.open(lq_path) as source:
            lq_image = source.convert("RGB")
        with Image.open(gt_path) as source:
            gt_image = source.convert("RGB")
        try:
            lq, gt = self.transform(lq_image, gt_image)
        except ValueError as exc:
            raise ValueError(f"Invalid image pair {lq_path}, {gt_path}: {exc}") from exc

        if self.is_denoise and self.sigma > 0:
            seed = None
            if not self.transform.is_train:
                seed = zlib.crc32(f"{self.task_name}:{real_index}".encode()) & 0x7FFFFFFF
            lq = add_gaussian_noise(lq, self.sigma, seed=seed)

        return {
            "lq": lq,
            "gt": gt,
            "task_name": self.task_name,
            "image_id": str(lq_path),
            "deg_type": self.deg_type,
        }


def _classification_labels(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        labels = list(value)
    else:
        raise ValueError(f"{field} must be a label or a list of labels")
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"{field} must contain non-empty string labels")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field} contains duplicate labels: {labels}")
    return labels


class ClassificationDataset(Dataset):
    """LQ images with multi-hot labels in an explicit, fixed taxonomy order."""

    def __init__(
        self,
        tasks: Sequence[dict[str, Any]],
        deg_types: Sequence[str],
        image_size: int,
        training: bool = True,
        hflip_prob: float = 0.5,
        preprocessing: dict[str, Any] | None = None,
        return_gt: bool = False,
    ) -> None:
        if isinstance(deg_types, str):
            raise ValueError("data.deg_types must be a list of labels")
        self.deg_types = _classification_labels(deg_types, "data.deg_types")
        self.num_deg_types = len(self.deg_types)
        self.image_size = image_size
        if image_size <= 0:
            raise ValueError("Classification image_size must be positive")
        self.training = training
        self.hflip_prob = hflip_prob
        self.return_gt = return_gt
        self.samples: list[dict[str, Any]] = []
        self.task_ranges: list[tuple[int, int]] = []
        self.task_weights: list[float] = []
        self.transforms: dict[str, PairedTransform] = {}
        task_names = set()

        for task in tasks:
            task_name = str(task["name"])
            if task_name in task_names:
                raise ValueError(f"Duplicate classification task name: {task_name}")
            task_names.add(task_name)
            options = {
                "image_size": image_size,
                "mode": "random_crop" if training else "center_crop",
                "interpolation": "bilinear",
                "exif_transpose": True,
                **(preprocessing or {}),
                **task.get("preprocessing", {}),
            }
            self.transforms[task_name] = PairedTransform(is_train=training, **options)
            if task.get("clean", False):
                if list(task["deg_type"]) != []:
                    raise ValueError("Clean tasks must explicitly use deg_type: []")
                labels = []
            else:
                labels = _classification_labels(task["deg_type"], f"Task {task_name} deg_type")
            self.task_weights.append(task_sampling_weight(task))
            unknown = set(labels) - set(self.deg_types)
            if unknown:
                raise ValueError(
                    f"Task {task_name} has unknown labels {sorted(unknown)}; known types: {self.deg_types}"
                )
            if task.get("repeat_ratio", 1) != 1:
                raise ValueError(
                    f"Classification task {task_name}: repeat_ratio must be 1; "
                    "use data.classification_sampling=task_balanced instead"
                )
            label = torch.zeros(self.num_deg_types)
            for deg_type in labels:
                label[self.deg_types.index(deg_type)] = 1.0

            is_synthetic_noise = "noise" in labels and (
                task.get("noise_sigma") is not None
                or (task.get("gt_path") is not None and task.get("gt_path") == task.get("lq_path"))
            )
            sigma = 0.0
            if is_synthetic_noise:
                if task.get("noise_sigma") is not None:
                    sigma = float(task["noise_sigma"])
                else:
                    sigma = parse_sigma_from_name(task_name) or 0.0
                if not math.isfinite(sigma) or sigma <= 0:
                    raise ValueError(
                        f"Denoise classification task {task_name} must end with _<sigma> or set noise_sigma"
                    )
            gt_paths = {}
            if return_gt:
                pairs = pair_images(task)
                image_paths = [lq for lq, _ in pairs]
                gt_paths = {lq: gt for lq, gt in pairs}
            elif task.get("manifest"):
                key = "gt" if is_synthetic_noise else "lq"
                image_paths = [row[key] for row in read_manifest(task, require_gt=is_synthetic_noise)]
            else:
                source_dir = Path(task["gt_path"] if is_synthetic_noise else task["lq_path"])
                if task.get("lq_subdir") and not is_synthetic_noise:
                    source_dir /= task["lq_subdir"]
                if task.get("gt_subdir") and is_synthetic_noise:
                    source_dir /= task["gt_subdir"]
                image_paths = list_images(source_dir, recursive=bool(task.get("recursive", False)))
            excluded = task.get("exclude_filenames", [])
            if (
                not isinstance(excluded, Sequence)
                or isinstance(excluded, (str, bytes, bytearray))
                or any(not isinstance(name, str) or not name for name in excluded)
            ):
                raise ValueError(f"Task {task_name} exclude_filenames must be a list of filenames")
            if len(excluded) != len(set(excluded)):
                raise ValueError(f"Task {task_name} exclude_filenames contains duplicate names")
            if excluded and not training:
                raise ValueError(f"Task {task_name} exclude_filenames is only allowed for training")
            missing = set(excluded) - {path.name for path in image_paths}
            if missing:
                raise ValueError(f"Task {task_name} excluded filenames do not exist: {sorted(missing)}")
            excluded = set(excluded)
            image_paths = [path for path in image_paths if path.name not in excluded]
            if not image_paths:
                raise ValueError(f"Task {task_name} is empty after exclude_filenames")
            start = len(self.samples)
            for image_path in image_paths:
                self.samples.append(
                    {
                        "path": image_path,
                        "gt_path": gt_paths.get(image_path),
                        "label": label,
                        "sigma": sigma,
                        "task_name": task_name,
                    }
                )
            self.task_ranges.append((start, len(self.samples)))
        if not self.samples:
            raise ValueError("Classification dataset must contain at least one task")

    def task_sampling_weights(self) -> torch.Tensor:
        """Each task has equal total probability, regardless of its image count."""
        weights = torch.empty(len(self), dtype=torch.double)
        for (start, end), weight in zip(self.task_ranges, self.task_weights, strict=True):
            weights[start:end] = weight / (end - start)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample["path"]) as source:
            image = source.convert("RGB")
        if self.return_gt:
            with Image.open(sample["gt_path"]) as source:
                gt_image = source.convert("RGB")
        else:
            gt_image = image
        image, gt_image = self.transforms[sample["task_name"]].geometry(image, gt_image)
        if self.training and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            gt_image = TF.hflip(gt_image)
        lq = TF.to_tensor(image) * 2.0 - 1.0
        gt = TF.to_tensor(gt_image) * 2.0 - 1.0
        if sample["sigma"] > 0:
            seed = None
            if not self.training:
                seed = zlib.crc32(f"{sample['task_name']}:{index}".encode()) & 0x7FFFFFFF
            lq = add_gaussian_noise(lq, sample["sigma"], seed=seed)
        result = {"lq": lq, "label": sample["label"]}
        if self.return_gt:
            result["gt"] = gt
        return result


class RoundRobinSampler(Sampler[int]):
    """One sample from each degradation group per batch (batch_size == num_groups)."""

    def __init__(self, group_boundaries: Sequence[tuple[int, int]]) -> None:
        self.group_boundaries = [tuple(b) for b in group_boundaries]
        self.num_groups = len(self.group_boundaries)
        self.group_sizes = [end - start for start, end in self.group_boundaries]
        if any(size <= 0 for size in self.group_sizes):
            raise ValueError(f"Round-robin groups must be non-empty, got {self.group_sizes}")

    def __iter__(self) -> Iterator[int]:
        per_group = []
        for start, end in self.group_boundaries:
            indices = list(range(start, end))
            random.shuffle(indices)
            per_group.append(indices)
        batch_count = min(self.group_sizes)
        batches = []
        for slot in range(batch_count):
            batches.extend(per_group[group][slot] for group in range(self.num_groups))
        return iter(batches)

    def __len__(self) -> int:
        return min(self.group_sizes) * self.num_groups


def _stack_images(batch: Sequence[dict[str, Any]], key: str) -> torch.Tensor:
    shapes = {tuple(item[key].shape) for item in batch}
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot batch variable image sizes {sorted(shapes)}. "
            "Use batch_size: 1 or fixed-size preprocessing."
        )
    return torch.stack([item[key] for item in batch])


def _collate_paired(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lq": _stack_images(batch, "lq"),
        "gt": _stack_images(batch, "gt"),
        "task_name": [item["task_name"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "deg_type": [item["deg_type"] for item in batch],
    }


def _collate_classification(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "lq": _stack_images(batch, "lq"),
        "label": torch.stack([item["label"] for item in batch]),
    }
    if "gt" in batch[0]:
        result["gt"] = _stack_images(batch, "gt")
    return result


def _tasks(cfg: OmegaConf, split: str) -> list[dict[str, Any]]:
    node = cfg.data.get(split)
    if node is None:
        return []
    return OmegaConf.to_container(node, resolve=True)


def build_loaders(
    cfg: OmegaConf, verbose: bool = True, eval_split: str = "test"
) -> tuple[DataLoader, OrderedDict[str, DataLoader]]:
    """Build train/test loaders for every stage.

    ``cfg.stage == "classifier"`` selects single-image classification loaders;
    the two restoration stages share the paired-image loader.
    """
    stage = str(cfg.stage)
    train_tasks = _tasks(cfg, "train")
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be val or test")
    test_tasks = _tasks(cfg, eval_split)
    if eval_split == "val" and not test_tasks:
        raise ValueError("Validation tasks are required when eval_split=val")
    num_workers = int(cfg.data.num_workers)
    persistent = bool(cfg.persistent_workers) and num_workers > 0
    pin_memory = bool(cfg.pin_memory)

    if not train_tasks:
        raise ValueError("cfg.data.train must contain at least one task")
    if not test_tasks and verbose:
        print("  [data] no test tasks; training will continue without periodic eval")

    task_groups = [train_tasks + test_tasks] if stage == "classifier" else [train_tasks, test_tasks]
    for tasks in task_groups:
        names = [str(task["name"]) for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate task names in train/evaluation tasks")
    test_loaders: OrderedDict[str, DataLoader] = OrderedDict()

    if stage == "classifier":
        if cfg.data.deg_types is None or isinstance(cfg.data.deg_types, str):
            raise ValueError("data.deg_types must be an explicit list of labels")
        deg_types = _classification_labels(list(cfg.data.deg_types), "data.deg_types")
        if len(deg_types) != int(cfg.model.num_deg_types):
            raise ValueError(
                f"model.num_deg_types={cfg.model.num_deg_types} does not match "
                f"data.deg_types ({len(deg_types)}): {deg_types}"
            )
        sampling = str(cfg.data.classification_sampling)
        classifier_options = OmegaConf.merge(_DEFAULTS.model.classifier, cfg.model.get("classifier", {}))
        patch_supervision = str(classifier_options.patch.target) != "none"
        if sampling not in {"uniform", "task_balanced"}:
            raise ValueError(
                f"data.classification_sampling must be 'uniform' or 'task_balanced', got {sampling!r}"
            )
        train_dataset = ClassificationDataset(
            train_tasks,
            deg_types,
            image_size=int(cfg.data.image_size),
            training=True,
            hflip_prob=float(cfg.data.augmentation.hflip_prob),
            preprocessing=preprocessing_options(cfg, "classifier_train", {}),
            return_gt=patch_supervision,
        )
        classification_sampler = None
        if sampling == "task_balanced":
            generator = torch.Generator()
            if cfg.seed is not None:
                generator.manual_seed(int(cfg.seed))
            else:
                generator.seed()
            classification_sampler = WeightedRandomSampler(
                train_dataset.task_sampling_weights(),
                num_samples=len(train_dataset),
                replacement=True,
                generator=generator,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.trainer.train_batch_size),
            shuffle=classification_sampler is None,
            sampler=classification_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=int(cfg.data.prefetch_factor) if num_workers else None,
            drop_last=True,
            collate_fn=_collate_classification,
        )
        for task in test_tasks:
            dataset = ClassificationDataset(
                [task],
                deg_types,
                image_size=int(cfg.data.image_size),
                training=False,
                preprocessing=preprocessing_options(cfg, "classifier_eval", {}),
                return_gt=patch_supervision,
            )
            test_loaders[str(task["name"])] = DataLoader(
                dataset,
                batch_size=int(cfg.eval.batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_classification,
            )
    else:
        aug = cfg.data.augmentation

        def train_transform(task):
            options = {"image_size": int(cfg.data.train_image_size)}
            options.update(preprocessing_options(cfg, "paired_train", task))
            return PairedTransform(
                **options,
                is_train=True,
                hflip_prob=float(aug.hflip_prob),
                vflip_prob=float(aug.vflip_prob),
                rot90_prob=float(aug.rot90_prob),
            )

        round_robin = bool(cfg.trainer.get("round_robin", False))
        if round_robin:
            # Reorder tasks by degradation group so each group is contiguous in
            # the ConcatDataset; this keeps RoundRobinSampler boundaries valid
            # for arbitrary YAML task orderings.
            tasks_by_deg: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
            for task in train_tasks:
                tasks_by_deg.setdefault(str(task["deg_type"]), []).append(task)
            if int(cfg.trainer.train_batch_size) != len(tasks_by_deg):
                raise ValueError(
                    "trainer.round_robin requires trainer.train_batch_size == number of "
                    f"degradation groups ({len(tasks_by_deg)}), got {int(cfg.trainer.train_batch_size)}"
                )
            grouped_tasks = [task for tasks in tasks_by_deg.values() for task in tasks]
            train_datasets = [PairedImageDataset(task, train_transform(task)) for task in grouped_tasks]
            unified = ConcatDataset(train_datasets)
            sizes_by_name = {dataset.task_name: len(dataset) for dataset in train_datasets}
            boundaries = []
            offset = 0
            for tasks in tasks_by_deg.values():
                size = sum(sizes_by_name[task["name"]] for task in tasks)
                boundaries.append((offset, offset + size))
                offset += size
            sampler: Sampler = RoundRobinSampler(boundaries)
            train_loader = DataLoader(
                unified,
                batch_size=int(cfg.trainer.train_batch_size),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=int(cfg.data.prefetch_factor) if num_workers else None,
                drop_last=True,
                collate_fn=_collate_paired,
            )
        else:
            train_datasets = [PairedImageDataset(task, train_transform(task)) for task in train_tasks]
            unified = ConcatDataset(train_datasets)
            paired_sampling = str(cfg.data.paired_sampling)
            if paired_sampling not in {"uniform", "task_balanced"}:
                raise ValueError("Invalid data.paired_sampling")
            sampler = None
            if paired_sampling == "task_balanced":
                weights = torch.cat(
                    [
                        torch.full(
                            (len(dataset),), task_sampling_weight(task) / len(dataset), dtype=torch.double
                        )
                        for task, dataset in zip(train_tasks, train_datasets, strict=True)
                    ]
                )
                sampler = WeightedRandomSampler(weights, len(unified), replacement=True)
            train_loader = DataLoader(
                unified,
                batch_size=int(cfg.trainer.train_batch_size),
                sampler=sampler,
                shuffle=sampler is None,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                prefetch_factor=int(cfg.data.prefetch_factor) if num_workers else None,
                drop_last=True,
                collate_fn=_collate_paired,
            )

        if verbose:
            print(f"  [data] train tasks={len(train_tasks)} samples={len(unified)}")

        for task in test_tasks:
            options = {"image_size": int(cfg.data.train_image_size)}
            options.update(preprocessing_options(cfg, "paired_eval", task))
            test_transform = PairedTransform(is_train=False, **options)
            dataset = PairedImageDataset(task, test_transform)
            test_loaders[str(task["name"])] = DataLoader(
                dataset,
                batch_size=int(cfg.eval.batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_paired,
            )
            if verbose:
                print(f"  [data] test {task['name']}: {len(dataset)} samples")

    return train_loader, test_loaders
