"""Geometry, manifest and fail-fast contracts for disk-backed datasets."""

import json
import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as TF

from sd_aio import data


def image_file(path, size=(80, 56)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.random.default_rng(17).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    image.save(path)
    return image


def manifest_task(tmp_path, suffix=".jsonl", *, clean=False):
    image_file(tmp_path / "images" / "a.png")
    image_file(tmp_path / "images" / "b.png")
    rows = [
        {"lq": "images/b.png", "gt": "images/b.png"},
        {"lq": "images/a.png", "gt": "images/a.png"},
    ]
    manifest = tmp_path / ("pairs" + suffix)
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) if suffix == ".jsonl" else json.dumps(rows)
    )
    return {
        "name": "train_haze",
        "deg_type": [] if clean else "haze",
        "clean": clean,
        "manifest": str(manifest),
    }


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_manifest_order_pairing_and_real_dataset_exclusions(tmp_path, suffix):
    task = manifest_task(tmp_path, suffix)
    task["exclude_filenames"] = ["a.png"]
    paired = data.PairedImageDataset(task, data.PairedTransform(32, is_train=False))
    assert len(paired) == 1
    assert paired.pairs[0][0].name == "b.png"
    assert torch.equal(paired[0]["lq"], paired[0]["gt"])
    classified = data.ClassificationDataset([task], ["haze"], 32)
    assert len(classified) == 1
    assert classified[0]["label"].tolist() == [1.0]


def test_manifest_clean_classification_without_gt(tmp_path):
    task = manifest_task(tmp_path, clean=True)
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"lq": "images/a.png"}) + "\n")
    classified = data.ClassificationDataset([task], ["haze", "rain"], 32, training=False)
    assert classified[0]["label"].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="gt path"):
        data.PairedImageDataset(task, data.PairedTransform(32))


def test_manifest_online_denoise_is_reproducible(tmp_path):
    task = manifest_task(tmp_path)
    task.update(name="eval_noise_25", deg_type="noise", noise_sigma=25)
    paired = data.PairedImageDataset(task, data.PairedTransform(32, is_train=False))
    first, second = paired[0], paired[0]
    assert torch.equal(first["lq"], second["lq"])
    assert not torch.equal(first["lq"], first["gt"])
    classified = data.ClassificationDataset([task], ["noise"], 32, training=False)
    assert torch.equal(classified[0]["lq"], classified[0]["lq"])


@pytest.mark.parametrize("case", ["empty", "duplicate", "missing", "missing_gt", "mixed_sources"])
def test_invalid_manifests_fail_at_construction(tmp_path, case):
    task = manifest_task(tmp_path)
    path = tmp_path / "pairs.jsonl"
    row = {"lq": "images/a.png", "gt": "images/a.png"}
    if case == "empty":
        path.write_text("")
    elif case == "duplicate":
        path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    elif case == "missing":
        path.write_text(json.dumps({**row, "lq": "absent.png"}))
    elif case == "missing_gt":
        path.write_text(json.dumps({"lq": "images/a.png"}))
    else:
        task["lq_path"] = str(tmp_path / "images")
    with pytest.raises((ValueError, FileNotFoundError)):
        data.PairedImageDataset(task, data.PairedTransform(32))


def test_recursive_relative_stem_is_unambiguous(tmp_path):
    task = {
        "name": "nested",
        "deg_type": "haze",
        "lq_path": str(tmp_path / "lq"),
        "gt_path": str(tmp_path / "gt"),
    }
    for side in ("lq", "gt"):
        for scene in ("scene1", "scene2"):
            image_file(tmp_path / side / scene / "same.png")
    task.update(recursive=True, pairing="relative_stem")
    pairs = data.pair_images(task)
    assert len(pairs) == 2
    assert all(lq.parent.name == gt.parent.name for lq, gt in pairs)
    task["pairing"] = "stem"
    with pytest.raises(ValueError, match="Ambiguous"):
        data.pair_images(task)


def test_ambiguous_gt_prefix_is_rejected(tmp_path):
    task = {
        "name": "prefix",
        "lq_path": str(tmp_path / "lq"),
        "gt_path": str(tmp_path / "gt"),
        "match_prefix_len": 4,
    }
    image_file(tmp_path / "lq" / "0001_hazy.png")
    image_file(tmp_path / "gt" / "0001.png")
    image_file(tmp_path / "gt" / "0001_other.png")
    with pytest.raises(ValueError, match="Ambiguous"):
        data.pair_images(task)


@pytest.mark.parametrize(
    "mode", ["random_crop", "center_crop", "resize_short_center_crop", "resize", "native"]
)
def test_geometry_is_synchronized_and_range_preserved(tmp_path, mode):
    image = image_file(tmp_path / "input.png")
    transform = data.PairedTransform(
        32, is_train=mode == "random_crop", mode=mode, hflip_prob=1, vflip_prob=1, rot90_prob=1
    )
    lq, gt = transform(image, image.copy())
    assert torch.equal(lq, gt)
    assert lq.shape == ((3, 56, 80) if mode == "native" else (3, 32, 32))
    assert -1 <= lq.min() <= lq.max() <= 1


def test_short_edge_crop_matches_explicit_bicubic_protocol(tmp_path):
    image = image_file(tmp_path / "input.png", size=(90, 60))
    transform = data.PairedTransform(32, is_train=False, mode="resize_short_center_crop")
    expected = TF.resize(image, (32, 48), TF.InterpolationMode.BICUBIC)
    expected = TF.to_tensor(TF.crop(expected, 0, 8, 32, 32)) * 2 - 1
    assert torch.equal(transform(image, image)[0], expected)


@pytest.mark.parametrize("size", [(80, 56), (20, 14)])
def test_legacy_crop_pixels_and_rng_sequence_are_unchanged(tmp_path, size):
    image = image_file(tmp_path / "input.png", size=size)
    random.seed(19)
    w, h = image.size
    legacy = image
    if min(h, w) < 32:
        scale = 32 / min(h, w)
        h, w = round(h * scale), round(w * scale)
        legacy = TF.resize(legacy, (h, w), TF.InterpolationMode.BICUBIC)
    top, left = random.randint(0, h - 32), random.randint(0, w - 32)
    legacy = TF.crop(legacy, top, left, 32, 32)
    if random.random() < 0.25:
        legacy = TF.hflip(legacy)
    if random.random() < 0.25:
        legacy = TF.vflip(legacy)
    if random.random() < 0.25:
        legacy = TF.rotate(legacy, random.choice([90, -90]))
    expected_next_random = random.random()
    expected = TF.to_tensor(legacy) * 2 - 1
    random.seed(19)
    actual = data.PairedTransform(32, hflip_prob=0.25, vflip_prob=0.25, rot90_prob=0.25)(image, image)[0]
    assert torch.equal(actual, expected)
    assert random.random() == expected_next_random


def test_misaligned_image_dimensions_fail_before_resize():
    transform = data.PairedTransform(32, mode="resize")
    with pytest.raises(ValueError, match="identical sizes"):
        transform(Image.new("RGB", (64, 32)), Image.new("RGB", (32, 64)))


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "invalid"},
        {"mode": "random_crop", "is_train": False},
        {"mode": "resize", "image_size": 0},
        {"interpolation": "invalid"},
        {"hflip_prob": 2},
    ],
)
def test_invalid_preprocessing_rejected(options):
    with pytest.raises(ValueError):
        data.PairedTransform(**{"image_size": 32, **options})


def test_task_preprocessing_overrides_shared_config():
    cfg = OmegaConf.create(
        {"data": {"preprocessing": {"paired_eval": {"mode": "native", "interpolation": "bicubic"}}}}
    )
    task = {"preprocessing": {"mode": "resize", "image_size": 48}}
    options = data.preprocessing_options(cfg, "paired_eval", task)
    assert options == {"mode": "resize", "image_size": 48, "interpolation": "bicubic"}
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        data.preprocessing_options(cfg, "paired_eval", {"preprocessing": {"typo": True}})


def test_manifest_loaders_support_task_transform_and_eval_geometry(tmp_path):
    task = manifest_task(tmp_path)
    task["preprocessing"] = {"mode": "resize", "image_size": 24}
    cfg = OmegaConf.create(
        {
            "stage": "vae_encoder",
            "pin_memory": False,
            "persistent_workers": False,
            "data": {
                "num_workers": 0,
                "train_image_size": 32,
                "paired_sampling": "uniform",
                "augmentation": {"hflip_prob": 0, "vflip_prob": 0, "rot90_prob": 0},
                "train": [task],
                "test": [
                    {
                        **task,
                        "name": "test_haze",
                        "preprocessing": {"mode": "resize_short_center_crop", "image_size": 40},
                    }
                ],
            },
            "trainer": {"train_batch_size": 2},
            "eval": {"batch_size": 2},
        }
    )
    train, test = data.build_loaders(cfg, verbose=False)
    assert next(iter(train))["lq"].shape == (2, 3, 24, 24)
    assert next(iter(test["test_haze"]))["gt"].shape == (2, 3, 40, 40)


def test_manifest_root_is_relative_to_manifest_location(tmp_path):
    images = tmp_path / "assets"
    image_file(images / "lq.png")
    image_file(images / "gt.png")
    manifest = tmp_path / "lists" / "train.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps([{"lq": "lq.png", "gt": "gt.png"}]))
    rows = data.read_manifest({"manifest": str(manifest), "manifest_root": "../assets"})
    assert rows == [{"lq": (images / "lq.png").resolve(), "gt": (images / "gt.png").resolve()}]


def test_exif_policy_is_explicit_and_consistent_for_pair():
    image = Image.new("RGB", (48, 32))
    exif = image.getexif()
    exif[274] = 6
    image.info["exif"] = exif.tobytes()
    native = data.PairedTransform(0, is_train=False, exif_transpose=False)
    oriented = data.PairedTransform(0, is_train=False, exif_transpose=True)
    assert native(image, image)[0].shape == (3, 32, 48)
    first, second = oriented(image, image)
    assert first.shape == (3, 48, 32)
    assert torch.equal(first, second)


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), -1])
def test_nonfinite_noise_is_rejected_for_both_dataset_kinds(tmp_path, sigma):
    task = manifest_task(tmp_path)
    task.update(name="noise", deg_type="noise", noise_sigma=sigma)
    with pytest.raises(ValueError):
        data.PairedImageDataset(task, data.PairedTransform(32))
    with pytest.raises(ValueError):
        data.ClassificationDataset([task], ["noise"], 32)


def test_variable_size_batch_has_actionable_error():
    batch = [{"lq": torch.zeros(3, 32, 48)}, {"lq": torch.zeros(3, 48, 32)}]
    with pytest.raises(ValueError, match="batch_size: 1"):
        data._collate_classification(batch)


def test_eval_noise_has_local_rng_and_exact_legacy_scale():
    image = torch.zeros(3, 16, 16)
    initial_state = torch.random.get_rng_state().clone()
    actual = data.add_gaussian_noise(image, sigma=25, seed=123)
    assert torch.equal(torch.random.get_rng_state(), initial_state)
    generator = torch.Generator().manual_seed(123)
    expected = (torch.randn(image.shape, generator=generator) * (25 / 127.5)).clamp(-1, 1)
    assert torch.equal(actual, expected)


def test_paired_task_name_may_repeat_across_train_and_eval(tmp_path):
    task = manifest_task(tmp_path)
    cfg = OmegaConf.create(
        {
            "stage": "vae_encoder",
            "pin_memory": False,
            "persistent_workers": False,
            "data": {
                "num_workers": 0,
                "train_image_size": 32,
                "paired_sampling": "uniform",
                "augmentation": {"hflip_prob": 0, "vflip_prob": 0, "rot90_prob": 0},
                "train": [task],
                "test": [task],
            },
            "trainer": {"train_batch_size": 1},
            "eval": {"batch_size": 1},
        }
    )
    _, test = data.build_loaders(cfg, verbose=False)
    assert list(test) == [task["name"]]
    cfg.data.test.append(task)
    with pytest.raises(ValueError, match="Duplicate task"):
        data.build_loaders(cfg, verbose=False)
