from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import RandomSampler, WeightedRandomSampler

from sd_aio import data

TAXONOMY = ["haze", "rain", "snow", "lowlight"]


def make_task(tmp_path, name, labels, count=1, size=(48, 48)):
    directory = tmp_path / name
    directory.mkdir()
    for index in range(count):
        Image.new("RGB", size, "white").save(directory / f"{index:03}.png")
    return {"name": name, "deg_type": labels, "lq_path": str(directory)}


def make_cfg(train, test=(), sampling="task_balanced", taxonomy=None, classes=4):
    return OmegaConf.create(
        {
            "stage": "classifier",
            "seed": 42,
            "pin_memory": False,
            "persistent_workers": False,
            "model": {"num_deg_types": classes},
            "data": {
                "deg_types": TAXONOMY if taxonomy is None else taxonomy,
                "classification_sampling": sampling,
                "image_size": 32,
                "num_workers": 0,
                "augmentation": {"hflip_prob": 0},
                "train": train,
                "test": list(test),
            },
            "trainer": {"train_batch_size": 2},
            "eval": {"batch_size": 2},
        }
    )


def test_multilabel_uses_fixed_taxonomy_and_only_reads_lq(tmp_path):
    task = make_task(tmp_path, "foundir_mix", ["lowlight", "haze"])
    task["gt_path"] = str(tmp_path / "intentionally_missing_gt")
    ds = data.ClassificationDataset([task], TAXONOMY, 32, training=False)
    assert ds[0]["label"].tolist() == [1, 0, 0, 1]
    assert ds[0]["lq"].shape == (3, 32, 32)


def test_single_label_and_eval_subset_keep_training_taxonomy_order(tmp_path):
    train = [
        make_task(tmp_path, "train_snow", "snow", count=2),
        make_task(tmp_path, "train_mix", ["rain", "haze"], count=2),
    ]
    test = [make_task(tmp_path, "eval_haze", ["haze"])]
    loader, eval_loaders = data.build_loaders(make_cfg(train, test), verbose=False)
    assert loader.dataset[0]["label"].tolist() == [0, 0, 1, 0]
    assert next(iter(eval_loaders["eval_haze"]))["label"].tolist() == [[1, 0, 0, 0]]


@pytest.mark.parametrize("size", [(30, 20), (20, 30)])
@pytest.mark.parametrize("training", [False, True])
def test_small_rectangle_resize_crop_does_not_introduce_black_pixels(tmp_path, monkeypatch, size, training):
    task = make_task(tmp_path, "small", ["haze"], size=size)
    # Exercise the furthest valid training crop, not only the centre.
    monkeypatch.setattr(data.random, "randint", lambda low, high: high)
    ds = data.ClassificationDataset([task], TAXONOMY, 32, training=training, hflip_prob=0)
    assert torch.equal(ds[0]["lq"], torch.ones(3, 32, 32))


@pytest.mark.parametrize("labels", [[], "", ["haze", "haze"], ["haze", "unknown"], ["haze", 7], None])
def test_bad_sample_labels_fail_fast(tmp_path, labels):
    task = make_task(tmp_path, "bad", labels)
    with pytest.raises(ValueError):
        data.ClassificationDataset([task], TAXONOMY, 32)


@pytest.mark.parametrize("taxonomy", [[], ["haze", "haze"], ["haze", ""], "haze"])
def test_bad_taxonomy_fails_fast(tmp_path, taxonomy):
    task = make_task(tmp_path, "bad_taxonomy", "haze")
    with pytest.raises(ValueError):
        data.ClassificationDataset([task], taxonomy, 32)


def test_model_class_count_must_match_taxonomy(tmp_path):
    task = make_task(tmp_path, "train", "haze", count=2)
    with pytest.raises(ValueError, match="num_deg_types"):
        data.build_loaders(make_cfg([task], classes=3), verbose=False)


def test_duplicate_task_names_fail_before_eval_loader_overwrite(tmp_path):
    first = make_task(tmp_path, "same_name", "haze", count=2)
    second = {**first, "deg_type": ["snow"]}
    with pytest.raises(ValueError, match="Duplicate"):
        data.ClassificationDataset([first, second], TAXONOMY, 32)
    with pytest.raises(ValueError, match="Duplicate"):
        data.build_loaders(make_cfg([first], [second]), verbose=False)


def test_task_balancing_uses_equal_probability_without_duplicated_samples(tmp_path):
    tasks = [
        make_task(tmp_path, "small_task", "snow", count=2),
        make_task(tmp_path, "large_task", ["haze", "rain"], count=6),
    ]
    loader, _ = data.build_loaders(make_cfg(tasks), verbose=False)
    ds = loader.dataset
    assert len(ds) == len(ds.samples) == 8
    assert len({sample["path"] for sample in ds.samples}) == 8
    assert isinstance(loader.sampler, WeightedRandomSampler)
    weights = loader.sampler.weights
    assert weights[:2].sum().item() == pytest.approx(weights[2:].sum().item())
    loader.sampler.num_samples = 6000
    draws = Counter(ds.samples[i]["task_name"] for i in loader.sampler)
    assert draws["small_task"] / 6000 == pytest.approx(0.5, abs=0.03)


def test_uniform_sampling_and_invalid_sampling_mode(tmp_path):
    task = make_task(tmp_path, "train", ["rain"], count=2)
    loader, _ = data.build_loaders(make_cfg([task], sampling="uniform"), verbose=False)
    assert isinstance(loader.sampler, RandomSampler)
    with pytest.raises(ValueError, match="classification_sampling"):
        data.build_loaders(make_cfg([task], sampling="surprise"), verbose=False)


def test_legacy_repetition_is_not_silently_applied_to_balanced_sampling(tmp_path):
    task = make_task(tmp_path, "train", ["haze"], count=2)
    task["repeat_ratio"] = 5
    with pytest.raises(ValueError, match="repeat_ratio"):
        data.ClassificationDataset([task], TAXONOMY, 32)


def test_training_exclusions_preserve_files_and_rebalance_remaining_samples(tmp_path):
    tasks = [
        make_task(tmp_path, "filtered_task", ["haze", "rain"], count=4),
        make_task(tmp_path, "other_task", "snow", count=6),
    ]
    tasks[0]["exclude_filenames"] = ["000.png", "002.png"]
    loader, _ = data.build_loaders(make_cfg(tasks), verbose=False)
    ds = loader.dataset
    assert len(ds.samples) == 8
    assert [sample["path"].name for sample in ds.samples[:2]] == ["001.png", "003.png"]
    assert (tmp_path / "filtered_task" / "000.png").is_file()
    assert (tmp_path / "filtered_task" / "002.png").is_file()
    assert loader.sampler.weights[:2].sum().item() == pytest.approx(loader.sampler.weights[2:].sum().item())


@pytest.mark.parametrize(
    "excluded",
    [["typo.png"], ["000.png", "000.png"], "000.png", [None], [""], ["000.png", "001.png"]],
)
def test_invalid_or_emptying_exclusions_fail_fast(tmp_path, excluded):
    task = make_task(tmp_path, "train", "haze", count=2)
    task["exclude_filenames"] = excluded
    with pytest.raises(ValueError):
        data.ClassificationDataset([task], TAXONOMY, 32)


def test_validation_images_cannot_be_excluded(tmp_path):
    task = make_task(tmp_path, "eval", "snow", count=2)
    task["exclude_filenames"] = ["000.png"]
    with pytest.raises(ValueError, match="only allowed for training"):
        data.ClassificationDataset([task], TAXONOMY, 32, training=False)
