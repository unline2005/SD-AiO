from omegaconf import OmegaConf

from sd_aio import data
from tests.helpers import make_synthetic_task


def test_classifier_validation_and_test_are_separate(tmp_path):
    train = make_synthetic_task(tmp_path, "train_haze", "haze", n_images=2)
    val = make_synthetic_task(tmp_path, "val_haze", "haze", n_images=1)
    test = make_synthetic_task(tmp_path, "test_haze", "haze", n_images=1)
    cfg = OmegaConf.create(
        {
            "stage": "classifier",
            "seed": 42,
            "pin_memory": False,
            "persistent_workers": False,
            "model": {"num_deg_types": 1},
            "data": {
                "deg_types": ["haze"],
                "classification_sampling": "uniform",
                "image_size": 32,
                "num_workers": 0,
                "prefetch_factor": 2,
                "augmentation": {"hflip_prob": 0.0},
                "train": [train],
                "val": [val],
                "test": [test],
            },
            "trainer": {"train_batch_size": 1},
            "eval": {"batch_size": 1},
        }
    )
    train_loader, val_loaders = data.build_loaders(cfg, verbose=False, eval_split="val")
    _, test_loaders = data.build_loaders(cfg, verbose=False, eval_split="test")
    assert list(val_loaders) == ["val_haze"]
    assert list(test_loaders) == ["test_haze"]
    train_paths = {s["path"] for s in train_loader.dataset.samples}
    val_paths = {s["path"] for s in val_loaders["val_haze"].dataset.samples}
    test_paths = {s["path"] for s in test_loaders["test_haze"].dataset.samples}
    assert train_paths.isdisjoint(val_paths | test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_missing_validation_fails_instead_of_using_test(tmp_path):
    import pytest

    task = make_synthetic_task(tmp_path, "train_haze", "haze", n_images=1)
    cfg = OmegaConf.create({"stage": "classifier", "data": {"train": [task], "val": []}})
    with pytest.raises(ValueError, match="Validation tasks"):
        data.build_loaders(cfg, eval_split="val")
