import pytest
import torch
from omegaconf import OmegaConf

from sd_aio.losses import pixel_loss
from sd_aio.optim import build_optimizer


@pytest.mark.parametrize("kind", ["adamw", "adam", "sgd"])
def test_optimizer_groups_and_roundtrip(kind):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    cfg = OmegaConf.create({"name": kind, "lr": 0.01})
    optimizer = build_optimizer([{"params": [parameter], "lr": 0.02}], cfg)
    parameter.square().sum().backward()
    optimizer.step()
    assert parameter.item() < 1.0
    restored = build_optimizer([parameter], cfg)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.param_groups[0]["lr"] == 0.02


def test_default_adamw_matches_torch():
    a = torch.nn.Parameter(torch.tensor([1.0]))
    b = torch.nn.Parameter(a.detach().clone())
    ours = build_optimizer([a], OmegaConf.create({"lr": 0.01}))
    native = torch.optim.AdamW([b], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for parameter, optimizer in ((a, ours), (b, native)):
        parameter.square().sum().backward()
        optimizer.step()
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("kind,expected", [("l1", 2.0), ("mse", 4.0)])
def test_pixel_loss_values_and_gradient(kind, expected):
    prediction = torch.tensor([2.0], requires_grad=True)
    value = pixel_loss(prediction, torch.zeros_like(prediction), kind, epsilon=1e-3)
    torch.testing.assert_close(value, torch.tensor(expected))
    value.backward()
    assert prediction.grad.item() > 0


def test_charbonnier_gradient_and_validation():
    prediction = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = pixel_loss(prediction, torch.zeros_like(prediction), "charbonnier", epsilon=1e-3)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0] == 0
    with pytest.raises(ValueError, match="epsilon"):
        pixel_loss(prediction, prediction, "charbonnier", epsilon=0)
    with pytest.raises(ValueError, match="shape"):
        pixel_loss(prediction, prediction[:1], "mse", epsilon=1e-3)
