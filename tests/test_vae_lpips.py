import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from sd_aio import checkpoint, vae_encoder
from tests.test_vae_pixel import make_cfg


class PerceptualProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 2, 1, bias=False).requires_grad_(False)
        self.target = None

    def forward(self, prediction, target):
        self.target = target.detach().clone()
        return (self.projection(prediction) - self.projection(target)).square().mean((1, 2, 3))


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("target", ["gt", "vae_reconstruction"])
def test_lpips_gradient_target_and_weighted_loss(tmp_path, monkeypatch, freeze, target):
    cfg = make_cfg(tmp_path, freeze, target)
    cfg.loss.lambda_pixel = 0.0
    cfg.loss.lambda_lpips = 0.2
    monkeypatch.setattr(vae_encoder.metrics, "load_lpips", lambda *args: PerceptualProbe().eval())
    model = vae_encoder.build_model(cfg)
    batch = {k: torch.rand(1, 3, 64, 64) * 2 - 1 for k in ("lq", "gt")}
    loss, logs = vae_encoder.compute_loss(model, model, batch, cfg)
    assert logs["loss"] == pytest.approx(0.2 * logs["loss_lpips"])
    with torch.no_grad():
        expected = (
            batch["gt"]
            if target == "gt"
            else model.frozen_vae.decode(vae_encoder._latent_mean(model, batch["gt"])).sample
        )
    torch.testing.assert_close(model.lpips.target, expected)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaln.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters()) is (not freeze)
    assert all(p.grad is None for p in model.lpips.parameters())
    optimizer = vae_encoder.make_optimizer(model, cfg)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in optimized for p in model.lpips.parameters())
    vae_encoder.set_train_mode(model)
    assert not model.lpips.training
    from safetensors.torch import load_file

    checkpoint.save_model_weights(model, tmp_path / "weights.safetensors")
    assert not any(k.startswith("lpips.") for k in load_file(tmp_path / "weights.safetensors"))


def test_disabled_lpips_does_not_load_and_old_resume_remains_valid(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)

    def forbidden(*args):
        raise AssertionError("LPIPS loaded with zero weight")

    monkeypatch.setattr(vae_encoder.metrics, "load_lpips", forbidden)
    model = vae_encoder.build_model(cfg)
    assert model.lpips is None
    out = tmp_path / "saved"
    out.mkdir()
    old = OmegaConf.create(OmegaConf.to_container(cfg))
    del old.loss.lambda_lpips
    del old.loss.lpips_net
    OmegaConf.save(old, out / "config.yaml")
    vae_encoder.validate_resume(cfg, out)
    cfg.loss.lambda_lpips = 0.1
    with pytest.raises(ValueError, match="loss"):
        vae_encoder.validate_resume(cfg, out)


def test_lpips_rejects_latent_target_and_invalid_weight(tmp_path):
    cfg = make_cfg(tmp_path, target="latent")
    cfg.loss.lambda_lpips = 0.1
    with pytest.raises(ValueError, match="pixel-space"):
        vae_encoder.build_model(cfg)
    cfg.loss.lambda_lpips = float("nan")
    with pytest.raises(ValueError, match="nonnegative"):
        vae_encoder.build_model(cfg)


def test_bf16_training_keeps_lpips_in_fp32(tmp_path, monkeypatch):
    class FloatProbe(PerceptualProbe):
        def forward(self, prediction, target):
            assert prediction.dtype == target.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            return super().forward(prediction, target)

    cfg = make_cfg(tmp_path, freeze=False)
    cfg.mixed_precision = "bf16"
    cfg.loss.lambda_lpips = 0.1
    monkeypatch.setattr(vae_encoder.metrics, "load_lpips", lambda *args: FloatProbe().eval())
    model = vae_encoder.build_model(cfg)
    batch = {k: torch.rand(1, 3, 64, 64) * 2 - 1 for k in ("lq", "gt")}
    loss, logs = vae_encoder.compute_loss(model, model, batch, cfg)
    loss.backward()
    assert logs["loss_lpips"] > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("freeze", [True, False])
def test_raw_gt_l1_plus_lpips_unit_weights(tmp_path, monkeypatch, freeze):
    cfg = make_cfg(tmp_path, freeze=freeze, target="gt")
    cfg.loss.pixel_type = "l1"
    cfg.loss.lambda_pixel = cfg.loss.lambda_lpips = 1.0
    monkeypatch.setattr(vae_encoder.metrics, "load_lpips", lambda *args: PerceptualProbe().eval())
    model = vae_encoder.build_model(cfg)
    batch = {k: torch.rand(1, 3, 64, 64) * 2 - 1 for k in ("lq", "gt")}
    loss, logs = vae_encoder.compute_loss(model, model, batch, cfg)
    pred, _ = vae_encoder._predict(model, model, batch["lq"])
    l1 = torch.nn.functional.l1_loss(pred, batch["gt"])
    perceptual = model.lpips(pred, batch["gt"]).mean()
    torch.testing.assert_close(loss, l1 + perceptual)
    assert logs["loss_pixel_l1"] == pytest.approx(l1.item())
    assert "loss_pixel_mse" not in logs
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaln.parameters())
