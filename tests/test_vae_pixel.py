from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import Dinov2Config, Dinov2Model

from sd_aio import checkpoint, classifier, config, vae_encoder
from tests.helpers import make_synthetic_task, make_tiny_vae
from tests.test_classifier_end_to_end import _launch_classifier


def make_cfg(tmp_path, freeze=True, target="vae_reconstruction"):
    sd = tmp_path / "sd"
    make_tiny_vae().save_pretrained(sd / "vae")
    dino = tmp_path / "dino"
    Dinov2Model(
        Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14)
    ).save_pretrained(dino)
    cls = classifier.DegradationClassifier(4, str(dino), freeze_encoder=False, head_hidden_dim=8)
    cls_path = tmp_path / "classifier.safetensors"
    checkpoint.save_model_weights(cls, cls_path)
    tasks = {
        s: [
            make_synthetic_task(tmp_path, name=f"{s}_{i}", deg_type="haze", n_images=4, image_size=64)
            for i in range(2)
        ]
        for s in ("train", "val", "test")
    }
    return OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH),
        {
            "stage": "vae_encoder",
            "output_dir": str(tmp_path / "out"),
            "seed": 5,
            "mixed_precision": "no",
            "pin_memory": False,
            "persistent_workers": False,
            "keep_last_checkpoints": 5,
            "model": {
                "sd_path": str(sd),
                "dino_path": str(dino),
                "num_deg_types": 4,
                "head_hidden_dim": 8,
                "cond_dim": 16,
                "adaln_layers": ["down2", "down3", "mid"],
                "degradation_classifier_path": str(cls_path),
                "pretrained_encoder_path": None,
                "train_deg_embedding": False,
                "vae_training": {"freeze_encoder": freeze, "target": target},
            },
            "data": {
                "tasks_file": None,
                **tasks,
                "paired_sampling": "task_balanced",
                "train_image_size": 64,
                "num_workers": 0,
                "augmentation": {"hflip_prob": 0, "vflip_prob": 0, "rot90_prob": 0},
            },
            "optimizer": {"lr": 0.001},
            "scheduler": {"name": "constant", "warmup_steps": 0},
            "loss": {"lambda_pixel": 1.0},
            "trainer": {
                "max_steps": 2,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "eval_split": "val",
                "eval_freq": 1,
                "eval_num_samples": 3,
                "num_images_save_eval": 4,
                "checkpointing_steps": 1,
                "log_every": 1,
            },
            "eval": {
                "batch_size": 2,
                "compute_lpips": False,
                "num_samples_per_task": 3,
                "pad_to_multiple": 8,
                "crop_to_multiple": 8,
            },
        },
    )


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("target", ["gt", "vae_reconstruction"])
def test_pixel_targets_gradients_and_condition_roundtrip(tmp_path, freeze, target):
    cfg = make_cfg(tmp_path, freeze, target)
    model = vae_encoder.build_model(cfg)
    batch = {"lq": torch.randn(1, 3, 64, 64).clamp(-1, 1), "gt": torch.randn(1, 3, 64, 64).clamp(-1, 1)}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    loss, _ = vae_encoder.compute_loss(model, model, batch, cfg)
    with torch.no_grad():
        pred, _ = vae_encoder._predict(model, model, batch["lq"])
        expected = (
            batch["gt"]
            if target == "gt"
            else model.frozen_vae.decode(vae_encoder._latent_mean(model, batch["gt"])).sample
        )
    torch.testing.assert_close(loss, torch.nn.functional.mse_loss(pred, expected))
    loss.backward()
    assert all(p.grad is None for p in model.frozen_vae.parameters())
    assert all(p.grad is None for p in model.deg_extractor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaln.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters()) is (not freeze)
    optimizer = vae_encoder.make_optimizer(model, cfg)
    optimizer.step()
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    folder = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path / "out")
    state = load_file(folder / "weights.safetensors")
    assert any(n.startswith("encoder.") for n in state) is (not freeze)
    assert all(n.startswith(("encoder.", "adaln.")) for n in state)
    torch.manual_seed(101)
    rebuilt = vae_encoder.build_model(cfg)
    assert not torch.equal(rebuilt.deg_extractor.deg_embedding, model.deg_extractor.deg_embedding)
    optimizer2 = vae_encoder.make_optimizer(rebuilt, cfg)
    step, _ = checkpoint.restore_training(rebuilt, optimizer2, None, folder)
    assert step == 1
    with torch.no_grad():
        before, _ = vae_encoder._predict(model, model, batch["lq"])
        after, _ = vae_encoder._predict(rebuilt, rebuilt, batch["lq"])
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    (folder / "weights.condition.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint.load_model_weights(rebuilt, folder / "weights.safetensors")


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("lpips_weight", [0.0, 0.1])
def test_pixel_two_cpu_ranks_eval_visualization_and_resume(tmp_path, freeze, lpips_weight):
    import json

    from PIL import Image

    cfg = make_cfg(tmp_path, freeze)
    cfg.loss.lambda_lpips = lpips_weight
    if lpips_weight > 0:
        cfg.eval.compute_lpips = True
        cfg.loss.lambda_lpips = 1.0
        cfg.loss.pixel_type = "l1"
        cfg.model.vae_training.target = "gt"
    file = tmp_path / "train.yaml"
    OmegaConf.save(cfg, file)
    first = _launch_classifier(file, tmp_path / "first.log")
    assert "Training finished at step 2" in first
    out = Path(cfg.output_dir)
    report = json.loads((out / "eval/metrics_step_00000001.json").read_text())
    if lpips_weight > 0:
        assert "lpips" in report["overall"]
    assert len(report["vis_paths"]) == 4
    for task in ("val_0", "val_1"):
        assert len(list((out / "eval").glob(f"step_00000001_{task}_*.png"))) == 2
    with Image.open(report["vis_paths"][0]) as im:
        assert im.size == (64 * 5, 64 + 28)
    cp = out / "checkpoints/checkpoint-00000002"
    aux = load_file(cp / "weights.condition.safetensors")
    resumed = _launch_classifier(
        file, tmp_path / "resume.log", ("trainer.max_steps=3", "trainer.resume_from=latest")
    )
    assert "at step 2" in resumed and "Training finished at step 3" in resumed
    saved = torch.load(out / "checkpoints/checkpoint-00000003/optimizer.pt", weights_only=True)
    assert saved["step"] == saved["scheduler"]["last_epoch"] == 3
    assert all(int(v["step"]) == 3 for v in saved["optimizer"]["state"].values())
    for n, v in load_file(out / "final/weights.condition.safetensors").items():
        torch.testing.assert_close(v, aux[n], rtol=0, atol=0)
    assert (out / "config.yaml").exists()
    cfg.model.vae_training.freeze_encoder = not freeze
    with pytest.raises(ValueError, match="model"):
        vae_encoder.validate_resume(cfg, cp)
