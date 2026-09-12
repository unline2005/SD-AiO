import pytest
import torch

from sd_aio import checkpoint, vae_encoder
from tests.test_vae_pixel import make_cfg


def test_random_encoder_freeze_and_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path, freeze=False, target="gt")
    cfg.model.vae_training.encoder_init = "random"
    torch.manual_seed(42)
    model = vae_encoder.build_model(cfg)
    assert not torch.equal(model.encoder.conv_in.weight, model.frozen_vae.encoder.conv_in.weight)
    assert model.encoder.state_dict().keys() == model.frozen_vae.encoder.state_dict().keys()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(("encoder.", "adaln."))
    frozen = {n: p.detach().clone() for n, p in model.frozen_vae.named_parameters()}
    batch = {"lq": torch.rand(1, 3, 64, 64) * 2 - 1, "gt": torch.rand(1, 3, 64, 64) * 2 - 1}
    loss, _ = vae_encoder.compute_loss(model, model, batch, cfg)
    loss.backward()
    assert model.encoder.conv_in.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaln.parameters())
    assert all(p.grad is None for p in model.frozen_vae.parameters())
    optimizer = vae_encoder.make_optimizer(model, cfg)
    optimizer.step()
    for n, p in model.frozen_vae.named_parameters():
        torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
    saved = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path / "run")
    torch.manual_seed(123)
    restored = vae_encoder.build_model(cfg)
    new_optimizer = vae_encoder.make_optimizer(restored, cfg)
    step, _ = checkpoint.restore_training(restored, new_optimizer, None, saved)
    assert step == 1
    with torch.no_grad():
        first, _ = vae_encoder._predict(model, model, batch["lq"])
        second, _ = vae_encoder._predict(restored, restored, batch["lq"])
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_random_frozen_encoder_rejected(tmp_path):
    cfg = make_cfg(tmp_path, freeze=True, target="gt")
    cfg.model.vae_training.encoder_init = "random"
    with pytest.raises(ValueError, match="must be trainable"):
        vae_encoder.build_model(cfg)
