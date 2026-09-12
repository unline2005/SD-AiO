from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from sd_aio import classifier, config
from sd_aio.data import ClassificationDataset, _collate_classification


def patch_cfg(tmp_path: Path, target: str) -> OmegaConf:
    dino = tmp_path / "dino"
    Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=42,
            patch_size=14,
        )
    ).save_pretrained(dino)
    return OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH),
        {
            "stage": "classifier",
            "model": {
                "dino_path": str(dino),
                "num_deg_types": 2,
                "freeze_encoder": False,
                "head_hidden_dim": 8,
                "classifier": {
                    "pad_to_patch_multiple": True,
                    "patch": {"target": target, "layers": [1, 2], "hidden_dim": 8},
                },
            },
            "optimizer": {"head_lr": 1e-3, "backbone_lr": 1e-4},
            "loss": {"classification": "bce", "patch": {"weight": 0.2, "mask_threshold": 0.05}},
        },
    )


@pytest.mark.parametrize(
    ("target", "shape"),
    [("mean_abs", (2, 1, 3, 3)), ("mask_fraction", (2, 1, 3, 3)), ("residual", (2, 3, 42, 42))],
)
def test_patch_heads_fuse_intermediate_blocks_and_backpropagate(tmp_path, target, shape):
    cfg = patch_cfg(tmp_path, target)
    model = classifier.build_model(cfg)
    batch = {
        "lq": torch.rand(2, 3, 29, 31) * 2 - 1,
        "gt": torch.rand(2, 3, 29, 31) * 2 - 1,
        "label": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    logits, prediction = model.forward_with_patch(batch["lq"])
    assert logits.shape == (2, 2)
    assert prediction.shape == shape
    loss, logs = classifier.compute_loss(model, model, batch, cfg)
    loss.backward()
    assert logs["patch_loss"] > 0
    assert model.patch_head.output.weight.grad is not None
    assert model.encoder.encoder.layer[0].attention.attention.query.weight.grad is not None


def test_patch_targets_have_expected_values():
    gt = torch.zeros(1, 3, 3, 3)
    lq = gt.clone()
    lq[:, :, :2, :2] = 1.0
    scalar_prediction = torch.zeros(1, 1, 2, 2)
    mean_loss = classifier._patch_loss(scalar_prediction, lq, gt, "mean_abs", 2, 0.25)
    torch.testing.assert_close(mean_loss, torch.tensor(0.125))
    mask_loss = classifier._patch_loss(scalar_prediction, lq, gt, "mask_fraction", 2, 0.25)
    torch.testing.assert_close(mask_loss, torch.tensor(0.6931472))
    residual_prediction = torch.zeros(1, 3, 4, 4)
    residual_loss = classifier._patch_loss(residual_prediction, lq, gt, "residual", 2, 0.25)
    torch.testing.assert_close(residual_loss, torch.tensor(2 / 9))


def test_classification_dataset_returns_aligned_pair_for_patch_supervision(tmp_path):
    lq_dir, gt_dir = tmp_path / "lq", tmp_path / "gt"
    lq_dir.mkdir()
    gt_dir.mkdir()
    clean = torch.zeros(3, 28, 28, dtype=torch.uint8)
    degraded = clean.clone()
    degraded[:, :14] = 255
    Image.fromarray(degraded.permute(1, 2, 0).numpy()).save(lq_dir / "a.png")
    Image.fromarray(clean.permute(1, 2, 0).numpy()).save(gt_dir / "a.png")
    dataset = ClassificationDataset(
        [{"name": "rain", "deg_type": ["rain"], "lq_path": str(lq_dir), "gt_path": str(gt_dir)}],
        ["rain"],
        image_size=28,
        training=False,
        hflip_prob=0,
        return_gt=True,
    )
    item = dataset[0]
    assert item["lq"].shape == item["gt"].shape == (3, 28, 28)
    assert not torch.equal(item["lq"], item["gt"])
    batch = _collate_classification([item])
    assert set(batch) == {"lq", "gt", "label"}


def test_patch_config_rejects_disabled_or_unknown_targets(tmp_path):
    base = {
        "stage": "classifier",
        "output_dir": str(tmp_path / "out"),
        "optimizer": {"head_lr": 1e-3, "backbone_lr": 1e-4},
        "trainer": {"train_batch_size": 1, "max_steps": 1, "eval_freq": 1, "checkpointing_steps": 1},
        "data": {"train": [{"name": "x"}]},
    }
    cfg = OmegaConf.merge(OmegaConf.load(config.DEFAULTS_PATH), base)
    cfg.model.classifier.patch.target = "mean_abs"
    with pytest.raises(ValueError, match="weight must be positive"):
        config.validate_training(cfg)
    cfg.model.classifier.patch.target = "typo"
    cfg.loss.patch.weight = 1.0
    with pytest.raises(ValueError, match="target must be"):
        config.validate_training(cfg)
