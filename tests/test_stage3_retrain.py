from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from sd_aio import config, spade
from tests.helpers import make_spade_cfg, make_synthetic_task, make_tiny_restorer, make_tiny_sd_repo
from tests.test_classifier_end_to_end import _launch_classifier


def test_residual_spade_is_identity_and_can_learn():
    module = spade.Spade(32, 32)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    c = torch.randn_like(x)
    result = module(x, c)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    result.square().mean().backward()
    assert module.gamma.weight.grad.abs().sum() > 0
    assert module.beta.weight.grad.abs().sum() > 0
    torch.testing.assert_close(x.grad, 2 * x.detach() / x.numel())


def test_decoder_training_output_is_not_clamped(monkeypatch):
    model = make_tiny_restorer("simple")
    monkeypatch.setattr(model.vae, "decode", lambda z: SimpleNamespace(sample=z[:, :3]))
    latent = torch.ones(1, 4, 8, 8, requires_grad=True)
    prediction = model.decode_latent(latent)
    assert prediction.min() > 1
    prediction.sum().backward()
    assert latent.grad.abs().sum() > 0


def test_eval_noise_is_sample_stable_without_changing_rng():
    model = make_tiny_restorer("simple")
    seeds = model.noise_seeds(["image-a", "image-b"])
    reference = torch.zeros(2, 4, 8, 8)
    state = torch.get_rng_state().clone()
    a = model.noise_like(reference, seeds)
    b = torch.cat([model.noise_like(reference[:1], [s]) for s in reversed(seeds)])
    torch.testing.assert_close(a, b.flip(0), rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())
    model.eval()
    images = torch.randn(1, 3, 64, 64)
    batch = dict(lq=images, gt=images, task_name=["Task"], image_id=["image-a"])
    with torch.no_grad():
        first = spade.eval_step(model, model, batch)["pred"]
        torch.randn(100)
        second = spade.eval_step(model, model, batch)["pred"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_stage3_two_cpu_ranks_val_cache_lora_eval_resume(tmp_path):
    import json

    from safetensors.torch import load_file

    sd = make_tiny_sd_repo(tmp_path)
    train = make_synthetic_task(tmp_path, "train_task", "haze", n_images=4)
    val = make_synthetic_task(tmp_path, "val_task", "haze", n_images=2)
    cfg = OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH), make_spade_cfg(tmp_path, train, tmp_path / "out", sd)
    )
    cfg.data.val = [val]
    cfg.trainer.eval_split = "val"
    cfg.trainer.max_steps = 2
    cfg.trainer.train_batch_size = 1
    cfg.trainer.gradient_accumulation_steps = 2
    cfg.trainer.eval_freq = 1
    cfg.trainer.eval_num_samples = 1
    cfg.trainer.checkpointing_steps = 1
    cfg.model.lora.unet_rank = 2
    cfg.model.lora.vae_encoder_rank = 2
    cfg.model.lora.vae_decoder_rank = 2
    cfg.model.lora.strategy = "only_attn"
    file = tmp_path / "stage3.yaml"
    OmegaConf.save(cfg, file)
    log = _launch_classifier(file, tmp_path / "first.log")
    assert "Training finished at step 2" in log
    out = Path(cfg.output_dir)
    weights = load_file(out / "final/weights.safetensors")
    assert any(k.startswith("unet.") and "lora" in k for k in weights)
    assert any("vae.decoder.up_blocks" in k and "lora" in k for k in weights)
    assert any("vae.encoder.down_blocks" in k and "lora" in k for k in weights)
    d = json.loads((out / "eval/metrics_step_00000001.json").read_text())
    assert "val_task" in d["task_metrics"] and d["vis_paths"]
    log = _launch_classifier(
        file, tmp_path / "resume.log", ("trainer.max_steps=3", "trainer.resume_from=latest")
    )
    assert "at step 2" in log and "Training finished at step 3" in log
    state = torch.load(out / "checkpoints/checkpoint-00000003/optimizer.pt", weights_only=True)
    assert state["step"] == state["scheduler"]["last_epoch"] == 3
    assert all(int(v["step"]) == 3 for v in state["optimizer"]["state"].values())
    cfg.model.spade_version = 1
    with pytest.raises(ValueError, match="architecture"):
        spade.validate_resume(cfg, out)
