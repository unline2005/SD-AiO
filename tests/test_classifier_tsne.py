from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from sd_aio.classifier import save_tsne_visualization


class TinyClassifier(torch.nn.Module):
    def eval(self):
        return self

    def forward_features(self, images):
        features = images.mean(dim=(-1, -2))
        return features, torch.zeros(images.shape[0], 2)


def test_tsne_features_are_fixed_per_task_and_sources_are_separate(tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {"deg_types": ["haze", "rain"]},
            "eval": {
                "tsne": {
                    "samples_per_task": 2,
                    "plot_python": "python",
                    "plot_script": "plot.py",
                    "perplexity": 30,
                    "seed": 42,
                    "iterations": 250,
                }
            },
        }
    )
    batch = {"lq": torch.randn(3, 3, 8, 8), "label": torch.tensor([[1, 0], [1, 0], [1, 0]])}
    loaders = {"val_FoundIR_haze": [batch], "val_GGT_haze": [batch], "val_CDD11_haze": [batch]}
    with patch("subprocess.run") as run:
        target = save_tsne_visualization(
            TinyClassifier(),
            loaders,
            cfg,
            device=torch.device("cpu"),
            weight_dtype=torch.float32,
            output_dir=tmp_path,
            step=10,
        )
    payload = np.load(target / "features.npz")
    assert payload["features"].shape == (6, 3)
    assert payload["sources"].tolist() == ["FoundIR", "FoundIR", "GGT", "GGT", "CDD11", "CDD11"]
    run.assert_called_once()
