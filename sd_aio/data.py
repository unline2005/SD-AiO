"""Data loading: pairing, online degradation, transforms and loaders.

Design rules
------------
* Every directory / pairing problem fails loudly at construction time.  There
  is no ``0 == 0`` silently-empty-dataset path.
* Denoise datasets are always clean-clean on disk and degraded online with
  ``sigma / 127.5`` in [-1, 1] tensor space.  Eval noise is deterministic per
  sample index (crc32), train noise is random.
* Test loaders always use ``num_workers=0`` for determinism.
"""

from __future__ import annotations

import random
import re
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"Image directory contains no supported images: {directory}")
    return files


def _subdir(task: dict[str, Any], key: str) -> Path | None:
    value = task.get(key)
    return Path(value) if value else None


def pair_images(task: dict[str, Any]) -> list[tuple[Path, Path]]:
    """Pair LQ and GT files by stem, prefix, or (optionally) sub-directory stem."""
    lq_root = Path(task["lq_path"])
    gt_root = Path(task["gt_path"])
    lq_sub = _subdir(task, "lq_subdir")
    gt_sub = _subdir(task, "gt_subdir")

    lq_dir = lq_root / lq_sub if lq_sub is not None else lq_root
    gt_dir = gt_root / gt_sub if gt_sub is not None else gt_root
    lq_files = list_images(lq_dir)
    gt_files = list_images(gt_dir)

    prefix_len = int(task.get("match_prefix_len", 0) or 0)
    if prefix_len > 0:
        gt_by_prefix = {p.stem[:prefix_len]: p for p in gt_files}
        pairs = []
        for lq in lq_files:
            gt = gt_by_prefix.get(lq.stem[:prefix_len])
            if gt is None:
                raise ValueError(
                    f"[{task.get('name', lq_root)}] no GT prefix match for {lq.name} "
                    f"(prefix_len={prefix_len})"
                )
            pairs.append((lq, gt))
        return pairs

    if (lq_sub is not None) != (gt_sub is not None):
        raise ValueError("lq_subdir and gt_subdir must be provided together")

    if len(lq_files) != len(gt_files):
        raise ValueError(f"[{task.get('name', lq_root)}] LQ ({len(lq_files)}) != GT ({len(gt_files)})")
    gt_by_stem = {p.stem: p for p in gt_files}
    pairs = []
    for lq in lq_files:
        gt = gt_by_stem.get(lq.stem)
        if gt is None:
            raise ValueError(f"[{task.get('name', lq_root)}] no GT image for {lq.name}")
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
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn_like(image, generator=generator) * (sigma / 127.5)
    else:
        noise = torch.randn_like(image) * (sigma / 127.5)
    return (image + noise).clamp(-1.0, 1.0)


def build_deg_types(task_entries: Sequence[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(task["deg_type"]) for task in task_entries))


class PairedTransform:
    """Identical geometric augmentation for LQ/GT pairs.

    Training always produces ``image_size x image_size`` crops; evaluation
    returns the full image so the shared eval path can apply the
    crop-to-multiple / pad-to-multiple protocol.
    """

    def __init__(
        self,
        image_size: int,
        is_train: bool = True,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rot90_prob: float = 0.0,
    ) -> None:
        self.image_size = image_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

    def __call__(self, lq_image: Image.Image, gt_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_train:
            return TF.to_tensor(lq_image) * 2.0 - 1.0, TF.to_tensor(gt_image) * 2.0 - 1.0
        lq_image, gt_image = self._crop_square(lq_image, gt_image)
        if random.random() < self.hflip_prob:
            lq_image, gt_image = TF.hflip(lq_image), TF.hflip(gt_image)
        if random.random() < self.vflip_prob:
            lq_image, gt_image = TF.vflip(lq_image), TF.vflip(gt_image)
        if random.random() < self.rot90_prob:
            angle = random.choice([90, -90])
            lq_image, gt_image = TF.rotate(lq_image, angle), TF.rotate(gt_image, angle)
        return TF.to_tensor(lq_image) * 2.0 - 1.0, TF.to_tensor(gt_image) * 2.0 - 1.0

    def _crop_square(self, lq_image: Image.Image, gt_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        w, h = gt_image.size
        target = self.image_size
        if min(h, w) < target:
            scale = target / min(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
            lq_image = TF.resize(lq_image, (new_h, new_w), TF.InterpolationMode.BICUBIC)
            gt_image = TF.resize(gt_image, (new_h, new_w), TF.InterpolationMode.BICUBIC)
            h, w = new_h, new_w
        top = random.randint(0, max(0, h - target))
        left = random.randint(0, max(0, w - target))
        return TF.crop(lq_image, top, left, target, target), TF.crop(gt_image, top, left, target, target)


class PairedImageDataset(Dataset):
    """One task: paired LQ/GT images with optional online Gaussian degradation."""

    def __init__(self, task: dict[str, Any], transform: PairedTransform) -> None:
        self.task_name = str(task["name"])
        self.deg_type = str(task["deg_type"])
        self.transform = transform
        self.repeat_ratio = max(1, int(task.get("repeat_ratio", 1) or 1))

        lq_path = Path(task["lq_path"])
        gt_path = Path(task["gt_path"])
        self.is_denoise = task.get("noise_sigma") is not None or (
            task.get("deg_type") == "noise" and lq_path == gt_path
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

        self.pairs = pair_images(task)
        self.indices = list(range(len(self.pairs))) * self.repeat_ratio

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        real_index = self.indices[index]
        lq_path, gt_path = self.pairs[real_index]
        lq_image = Image.open(lq_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        lq, gt = self.transform(lq_image, gt_image)

        if self.is_denoise and self.sigma > 0:
            seed = None
            if not self.transform.is_train:
                seed = zlib.crc32(f"{self.task_name}:{real_index}".encode()) & 0x7FFFFFFF
            lq = add_gaussian_noise(lq, self.sigma, seed=seed)

        return {
            "lq": lq,
            "gt": gt,
            "task_name": self.task_name,
            "deg_type": self.deg_type,
        }


class ClassificationDataset(Dataset):
    """Single-image dataset for the degradation classifier.

    Rain/haze samples read already-degraded LQ images; denoise samples read the
    clean directory and degrade online, so the classifier sees the same data
    distribution as the restoration stages.
    """

    def __init__(
        self,
        tasks: Sequence[dict[str, Any]],
        deg_types: Sequence[str],
        image_size: int,
        training: bool = True,
        hflip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.hflip_prob = hflip_prob
        self.deg_types = list(deg_types)
        self.num_deg_types = len(self.deg_types)
        self.samples: list[dict[str, Any]] = []

        for task in tasks:
            deg_type = str(task["deg_type"])
            if deg_type not in self.deg_types:
                raise ValueError(
                    f"deg_type {deg_type} of task {task.get('name')} is not in known types {self.deg_types}"
                )
            label = torch.zeros(self.num_deg_types)
            label[self.deg_types.index(deg_type)] = 1.0
            repeat = max(1, int(task.get("repeat_ratio", 1) or 1))

            is_synthetic_noise = deg_type == "noise" and (
                task.get("noise_sigma") is not None or task.get("gt_path") == task.get("lq_path")
            )
            sigma = 0.0
            if is_synthetic_noise:
                if task.get("noise_sigma") is not None:
                    sigma = float(task["noise_sigma"])
                else:
                    sigma = parse_sigma_from_name(str(task["name"])) or 0.0
                if sigma <= 0:
                    raise ValueError(
                        f"Denoise classification task {task.get('name')} must end with "
                        "_<sigma> or set noise_sigma"
                    )
            source_dir = Path(task["gt_path"] if is_synthetic_noise else task["lq_path"])
            for image_path in list_images(source_dir):
                for _ in range(repeat):
                    self.samples.append(
                        {
                            "path": image_path,
                            "label": label,
                            "sigma": sigma,
                            "task_name": str(task["name"]),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        w, h = image.size
        target = self.image_size
        if min(h, w) < target:
            scale = target / min(h, w)
            image = TF.resize(image, (round(h * scale), round(w * scale)))
            h, w = image.size
        if self.training:
            top, left = (
                random.randint(0, max(0, h - target)),
                random.randint(0, max(0, w - target)),
            )
        else:
            top, left = (h - target) // 2, (w - target) // 2
        image = TF.crop(image, top, left, target, target)
        if self.training and random.random() < self.hflip_prob:
            image = TF.hflip(image)
        lq = TF.to_tensor(image) * 2.0 - 1.0
        if sample["sigma"] > 0:
            seed = None
            if not self.training:
                seed = zlib.crc32(f"{sample['task_name']}:{index}".encode()) & 0x7FFFFFFF
            lq = add_gaussian_noise(lq, sample["sigma"], seed=seed)
        return {"lq": lq, "label": sample["label"]}


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


def _collate_paired(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lq": torch.stack([item["lq"] for item in batch]),
        "gt": torch.stack([item["gt"] for item in batch]),
        "task_name": [item["task_name"] for item in batch],
        "deg_type": [item["deg_type"] for item in batch],
    }


def _collate_classification(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lq": torch.stack([item["lq"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def _tasks(cfg: OmegaConf, split: str) -> list[dict[str, Any]]:
    node = cfg.data.get(split)
    if node is None:
        return []
    return OmegaConf.to_container(node, resolve=True)


def build_loaders(cfg: OmegaConf, verbose: bool = True) -> tuple[DataLoader, OrderedDict[str, DataLoader]]:
    """Build train/test loaders for every stage.

    ``cfg.stage == "classifier"`` selects single-image classification loaders;
    the two restoration stages share the paired-image loader.
    """
    stage = str(cfg.stage)
    train_tasks = _tasks(cfg, "train")
    test_tasks = _tasks(cfg, "test")
    num_workers = int(cfg.data.num_workers)
    persistent = bool(cfg.persistent_workers) and num_workers > 0
    pin_memory = bool(cfg.pin_memory)

    if not train_tasks:
        raise ValueError("cfg.data.train must contain at least one task")
    if not test_tasks and verbose:
        print("  [data] no test tasks; training will continue without periodic eval")

    test_loaders: OrderedDict[str, DataLoader] = OrderedDict()

    if stage == "classifier":
        deg_types = build_deg_types(train_tasks)
        for task in test_tasks:
            if task["deg_type"] not in deg_types:
                raise ValueError(
                    f"Test deg_type {task['deg_type']} is not present in train deg_types {deg_types}"
                )
        train_dataset: Dataset = ClassificationDataset(
            train_tasks,
            deg_types,
            image_size=int(cfg.data.image_size),
            training=True,
            hflip_prob=float(cfg.data.augmentation.hflip_prob),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.trainer.train_batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            drop_last=True,
            collate_fn=_collate_classification,
        )
        for task in test_tasks:
            dataset = ClassificationDataset(
                [task],
                deg_types,
                image_size=int(cfg.data.image_size),
                training=False,
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
        train_transform = PairedTransform(
            image_size=int(cfg.data.train_image_size),
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
            train_datasets = [PairedImageDataset(task, train_transform) for task in grouped_tasks]
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
                drop_last=True,
                collate_fn=_collate_paired,
            )
        else:
            train_datasets = [PairedImageDataset(task, train_transform) for task in train_tasks]
            unified = ConcatDataset(train_datasets)
            train_loader = DataLoader(
                unified,
                batch_size=int(cfg.trainer.train_batch_size),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
                drop_last=True,
                collate_fn=_collate_paired,
            )

        if verbose:
            print(f"  [data] train tasks={len(train_tasks)} samples={len(unified)}")

        test_transform = PairedTransform(image_size=0, is_train=False)
        for task in test_tasks:
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
