from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf

from sd_aio import data
from tests.helpers import make_synthetic_task


def test_pair_images_by_prefix_and_stem(tmp_path):
    prefix_task = make_synthetic_task(tmp_path, "Train_Prefix", "haze", n_images=3)
    prefix_task["match_prefix_len"] = 4
    pairs = data.pair_images(prefix_task)
    assert len(pairs) == 3

    stem_task = make_synthetic_task(tmp_path, "Train_Stem", "haze", n_images=3)
    pairs = data.pair_images(stem_task)
    assert [a.stem for a, _ in pairs] == [b.stem for _, b in pairs]


def test_paired_dataset_online_noise_is_sigma_scaled_and_deterministic(tmp_path):
    task = make_synthetic_task(tmp_path, "Train_Denoise_15", "noise", n_images=4)
    transform = data.PairedTransform(image_size=32, is_train=True, hflip_prob=0.0)
    dataset = data.PairedImageDataset(task, transform)
    sample = dataset[0]
    assert sample["lq"].shape == (3, 32, 32)
    assert dataset.sigma == 15.0

    eval_transform = data.PairedTransform(image_size=0, is_train=False)
    eval_dataset = data.PairedImageDataset(task, eval_transform)
    first = eval_dataset[0]["lq"]
    second = eval_dataset[0]["lq"]
    assert torch.equal(first, second)  # crc32-seeded deterministic eval noise


def test_denoise_task_without_sigma_fails(tmp_path):
    task = make_synthetic_task(tmp_path, "Train_Denoise", "noise", n_images=2)
    with pytest.raises(ValueError):
        data.PairedImageDataset(task, data.PairedTransform(32, is_train=True))


def test_missing_directory_fails_fast(tmp_path):
    task = make_synthetic_task(tmp_path, "Train_X", "noise", n_images=1)
    task["lq_path"] = str(tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError):
        data.PairedImageDataset(task, data.PairedTransform(32, is_train=True))


def test_classification_dataset_synthesises_noise_for_denoise(tmp_path):
    train_tasks = [
        make_synthetic_task(tmp_path, "Train_Haze", "haze", n_images=2),
        make_synthetic_task(tmp_path, "Train_Denoise_25", "noise", n_images=2),
    ]
    dataset = data.ClassificationDataset(train_tasks, ["haze", "noise"], image_size=32, training=False)
    labels = [item["label"].argmax().item() for item in (dataset[i] for i in range(len(dataset)))]
    assert Counter(labels) == {0: 2, 1: 2}
    noise_samples = [dataset[i]["lq"] for i in range(2, 4)]
    assert bool((noise_samples[0] - noise_samples[1]).abs().sum() > 0)


def test_round_robin_sampler_one_sample_per_group(tmp_path):
    tasks = [
        make_synthetic_task(tmp_path, "Train_Haze", "haze", n_images=4),
        make_synthetic_task(tmp_path, "Train_Rain", "rain", n_images=4),
        make_synthetic_task(tmp_path, "Train_Denoise_15", "noise", n_images=4),
    ]
    cfg = OmegaConf.create(
        {
            "stage": "vae_encoder",
            "mixed_precision": "no",
            "pin_memory": False,
            "persistent_workers": False,
            "data": {
                "num_workers": 0,
                "train_image_size": 32,
                "augmentation": {
                    "hflip_prob": 0.0,
                    "vflip_prob": 0.0,
                    "rot90_prob": 0.0,
                },
                "train": tasks,
                "test": [],
            },
            "trainer": {"train_batch_size": 3, "round_robin": True},
            "eval": {"batch_size": 1},
        }
    )
    loader, _ = data.build_loaders(cfg, verbose=False)
    assert len(loader) == 4
    batch = next(iter(loader))
    assert sorted(batch["deg_type"]) == ["haze", "noise", "rain"]


def test_round_robin_batch_size_mismatch_fails(tmp_path):
    tasks = [
        make_synthetic_task(tmp_path, "Train_Haze", "haze", n_images=2),
        make_synthetic_task(tmp_path, "Train_Rain", "rain", n_images=2),
        make_synthetic_task(tmp_path, "Train_Denoise_15", "noise", n_images=2),
    ]
    cfg = OmegaConf.create(
        {
            "stage": "vae_encoder",
            "mixed_precision": "no",
            "pin_memory": False,
            "persistent_workers": False,
            "data": {
                "num_workers": 0,
                "train_image_size": 32,
                "augmentation": {
                    "hflip_prob": 0.0,
                    "vflip_prob": 0.0,
                    "rot90_prob": 0.0,
                },
                "train": tasks,
                "test": [],
            },
            "trainer": {"train_batch_size": 2, "round_robin": True},
            "eval": {"batch_size": 1},
        }
    )
    with pytest.raises(ValueError, match="number of degradation groups"):
        data.build_loaders(cfg, verbose=False)
