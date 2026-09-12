import pytest
import torch
from omegaconf import OmegaConf

from sd_aio import config, data
from tests.helpers import make_synthetic_task


@pytest.mark.parametrize("stage", ["classifier", "vae_encoder", "spade"])
def test_three_source_probability_mass(stage, tmp_path):
    tasks = []
    for source, count in enumerate([2, 3, 4]):
        for index in range(count):
            task = make_synthetic_task(tmp_path, f"source{source}_task{index}", "haze", n_images=index + 1)
            task["sampling_weight"] = 1 / count
            tasks.append(task)
    cfg = OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH),
        {
            "stage": stage,
            "model": {"num_deg_types": 4},
            "persistent_workers": False,
            "data": {
                "train": tasks,
                "test": [],
                "num_workers": 0,
                "train_image_size": 32,
                "image_size": 32,
                "deg_types": ["haze", "rain", "snow", "lowlight"],
                "paired_sampling": "task_balanced",
                "classification_sampling": "task_balanced",
            },
            "trainer": {"train_batch_size": 1},
        },
    )
    loader, _ = data.build_loaders(cfg, verbose=False)
    weights = loader.sampler.weights
    offset = 0
    for count in [2, 3, 4]:
        size = sum(range(1, count + 1))
        torch.testing.assert_close(
            weights[offset : offset + size].sum() / weights.sum(), torch.tensor(1 / 3, dtype=torch.double)
        )
        offset += size


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_task_sampling_weights_fail(value):
    with pytest.raises(ValueError, match="positive and finite"):
        data.task_sampling_weight({"sampling_weight": value})
