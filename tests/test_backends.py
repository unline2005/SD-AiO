from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler, FluxPipeline
from omegaconf import OmegaConf

from sd_aio.backends import (
    LatentAffine,
    diffusion_x0,
    flow_x0,
    pack_flux_latents,
    require_backend,
    unpack_flux_latents,
    validate_sd_unet_configs,
)


@pytest.mark.parametrize("scale,shift", [(0.18215, 0.0), (1.5305, 0.0609), (0.3611, 0.1159)])
def test_latent_affine_roundtrip_and_gradients(scale, shift):
    affine = LatentAffine(scale, shift)
    latent = torch.randn(2, 16, 8, 10, dtype=torch.float64, requires_grad=True)
    encoded = affine.encode(latent)
    torch.testing.assert_close(encoded, (latent - shift) * scale)
    restored = affine.decode(encoded)
    torch.testing.assert_close(restored, latent)
    restored.sum().backward()
    torch.testing.assert_close(latent.grad, torch.ones_like(latent))


@pytest.mark.parametrize("scale,shift", [(0, 0), (-1, 0), (float("nan"), 0), (1, float("inf"))])
def test_invalid_latent_affine_fails(scale, shift):
    with pytest.raises(ValueError, match="finite"):
        LatentAffine(scale, shift)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction", "sample"])
def test_diffusion_clean_prediction_matches_diffusers_and_keeps_gradients(prediction_type):
    scheduler = DDPMScheduler(num_train_timesteps=100, prediction_type=prediction_type, clip_sample=False)
    clean = torch.randn(2, 4, 6, 8)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([10, 90])
    alpha = scheduler.alphas_cumprod[timesteps]
    a = alpha.reshape(-1, 1, 1, 1)
    noisy = scheduler.add_noise(clean, noise, timesteps)
    target = {
        "epsilon": noise,
        "v_prediction": a.sqrt() * noise - (1 - a).sqrt() * clean,
        "sample": clean,
    }[prediction_type]
    prediction = target.detach().clone().requires_grad_()
    result = diffusion_x0(noisy, prediction, alpha, prediction_type)
    torch.testing.assert_close(result, clean, atol=1e-6, rtol=1e-5)
    expected = torch.cat(
        [
            scheduler.step(prediction[i : i + 1], int(t), noisy[i : i + 1]).pred_original_sample
            for i, t in enumerate(timesteps)
        ]
    )
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    result.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_flow_uses_noise_minus_clean_velocity_not_diffusion_v():
    clean = torch.randn(3, 16, 8, 12)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.0, 0.4, 1.0])
    s = sigma.reshape(-1, 1, 1, 1)
    noisy = (1 - s) * clean + s * noise
    velocity = (noise - clean).requires_grad_()
    recovered = flow_x0(noisy, velocity, sigma)
    torch.testing.assert_close(recovered, clean, atol=3e-7, rtol=1e-5)
    recovered.sum().backward()
    torch.testing.assert_close(velocity.grad, -s.expand_as(velocity))
    packed_result = flow_x0(pack_flux_latents(noisy), pack_flux_latents(velocity), sigma)
    torch.testing.assert_close(unpack_flux_latents(packed_result, 8, 12), clean, atol=3e-7, rtol=1e-5)


def test_prediction_shapes_and_conventions_fail_fast():
    latent = torch.zeros(2, 4, 8, 8)
    with pytest.raises(ValueError, match="per batch"):
        diffusion_x0(latent, latent, torch.ones(3), "epsilon")
    with pytest.raises(ValueError, match="shapes"):
        flow_x0(latent, latent[:1], 0.5)
    with pytest.raises(ValueError, match="Unknown diffusion"):
        diffusion_x0(latent, latent, 0.5, "flow")


@pytest.mark.parametrize("transpose", [False, True])
def test_flux_pack_matches_diffusers_patch_order_and_roundtrips(transpose):
    latent = torch.arange(2 * 3 * 6 * 8).reshape(2, 3, 6, 8).float()
    if transpose:
        latent = latent.transpose(2, 3)
    latent.requires_grad_()
    batch, channels, height, width = latent.shape
    packed = pack_flux_latents(latent)
    expected = FluxPipeline._pack_latents(latent.contiguous(), batch, channels, height, width)
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
    torch.testing.assert_close(packed[0, 0], latent[0, :, :2, :2].reshape(-1), atol=0, rtol=0)
    restored = unpack_flux_latents(packed, height, width)
    official = FluxPipeline._unpack_latents(packed, height * 8, width * 8, 8)
    torch.testing.assert_close(restored, official, atol=0, rtol=0)
    torch.testing.assert_close(restored, latent, atol=0, rtol=0)
    restored.square().sum().backward()
    torch.testing.assert_close(latent.grad, 2 * latent)


def test_flux_rejects_implicit_geometry_truncation():
    with pytest.raises(ValueError, match="even"):
        pack_flux_latents(torch.zeros(1, 16, 7, 8))
    with pytest.raises(ValueError, match="match"):
        unpack_flux_latents(torch.zeros(1, 15, 64), 8, 8)
    with pytest.raises(ValueError, match="even"):
        unpack_flux_latents(torch.zeros(1, 16, 64), 7, 8)


def _sd_configs():
    return [
        dict(
            _class_name="UNet2DConditionModel",
            in_channels=4,
            out_channels=4,
            block_out_channels=[320, 640, 1280, 1280],
            cross_attention_dim=1024,
        ),
        dict(_class_name="AutoencoderKL", latent_channels=4, block_out_channels=[128, 256, 512, 512]),
        dict(model_type="clip_text_model", hidden_size=1024),
        dict(_class_name="EulerDiscreteScheduler", prediction_type="epsilon"),
    ]


def test_sd_turbo_shapes_are_compatible_with_restoration_components():
    require_backend("sd_unet")
    validate_sd_unet_configs(*_sd_configs(), condition_channels=[320, 640, 1280], text_dim=1024)


@pytest.mark.parametrize("backend", ["sd3", "flux", "typo"])
def test_unimplemented_adapters_are_not_silently_accepted(backend):
    with pytest.raises(NotImplementedError, match="no restoration adapter"):
        require_backend(backend)


@pytest.mark.parametrize(
    "component,key,value,message",
    [
        (0, "_class_name", "SD3Transformer2DModel", "UNet2DConditionModel"),
        (0, "in_channels", 9, "channels"),
        (0, "addition_embed_type", "text_time", "Additional"),
        (0, "cross_attention_dim", 2048, "hidden_size"),
        (1, "latent_channels", 16, "channels"),
        (3, "_class_name", "FlowMatchEulerDiscreteScheduler", "Flow schedulers"),
    ],
)
def test_incompatible_components_are_rejected(component, key, value, message):
    configs = deepcopy(_sd_configs())
    configs[component][key] = value
    with pytest.raises(ValueError, match=message):
        validate_sd_unet_configs(*configs, condition_channels=[320, 640, 1280], text_dim=1024)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction", "sample"])
def test_stage3_all_prediction_types_train_and_evaluate_from_local_components(tmp_path, prediction_type):
    from sd_aio import config, spade
    from tests.helpers import make_spade_cfg, make_synthetic_task, make_tiny_sd_repo

    sd_path = make_tiny_sd_repo(tmp_path)
    scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    scheduler.register_to_config(prediction_type=prediction_type)
    scheduler.save_pretrained(sd_path / "scheduler")
    task = make_synthetic_task(tmp_path, "train_task", "haze", n_images=1)
    cfg = OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH), make_spade_cfg(tmp_path, task, tmp_path / "out", sd_path)
    )
    model = spade.build_model(cfg, device=torch.device("cpu"))
    assert model.prediction_type == prediction_type
    image = torch.randn(1, 3, 64, 64)
    batch = dict(lq=image, gt=image, task_name=["train_task"], image_id=["test.png"])
    loss, _ = spade.compute_loss(model, model, batch, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.condition_module.wrappers[0].spade.gamma.weight.grad.abs().sum() > 0
    result = spade.eval_step(model, model, batch)["pred"]
    assert result.shape == image.shape and torch.isfinite(result).all()
    assert result.min() >= -1 and result.max() <= 1


def test_existing_epsilon_forward_keeps_exact_operation_order(monkeypatch):
    from tests.helpers import make_tiny_restorer

    model = make_tiny_restorer("simple")
    model.condition_module = None
    latent = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latent)
    prediction = torch.randn_like(latent)
    monkeypatch.setattr(model, "encode_lq", lambda *args, **kwargs: latent)
    monkeypatch.setattr(model, "noise_like", lambda *args, **kwargs: noise)
    monkeypatch.setattr(model.unet, "forward", lambda *args, **kwargs: SimpleNamespace(sample=prediction))
    monkeypatch.setattr(model, "decode_latent", lambda value: value)
    timesteps = torch.full((2,), 10, dtype=torch.long)
    expected = latent + model._x0_coeff(timesteps).to(latent.dtype) * (noise - prediction)
    actual = model(torch.zeros(2, 3, 64, 64), torch.zeros(2, 1, 16), timestep=10)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_backend_rejection_happens_before_model_loading(monkeypatch):
    from sd_aio import config, spade

    cfg = OmegaConf.load(config.DEFAULTS_PATH)
    cfg.model.backend = "flux"
    cfg.model.timestep = 10
    cfg.model.sd_path = "not-loaded"
    cfg.loss.timestep = {"strategy": "fixed", "value": 10}
    monkeypatch.setattr(
        spade.UNet2DConditionModel,
        "load_config",
        lambda *args, **kwargs: pytest.fail("Unsupported backend attempted to load components"),
    )
    with pytest.raises(NotImplementedError, match="no restoration adapter"):
        spade.build_model(cfg)


def test_component_mismatch_fails_before_loading_weights(tmp_path, monkeypatch):
    import json

    from sd_aio import config, spade
    from tests.helpers import make_spade_cfg, make_synthetic_task, make_tiny_sd_repo

    root = make_tiny_sd_repo(tmp_path)
    component = root / "unet/config.json"
    metadata = json.loads(component.read_text())
    metadata["in_channels"] = 9
    component.write_text(json.dumps(metadata))
    task = make_synthetic_task(tmp_path, "train_task", "haze", n_images=1)
    cfg = OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH), make_spade_cfg(tmp_path, task, tmp_path / "out", root)
    )
    monkeypatch.setattr(
        spade.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Incompatible components attempted to load weights"),
    )
    with pytest.raises(ValueError, match="channels"):
        spade.build_model(cfg)


def test_stage3_resume_accepts_added_backend_default_but_rejects_architecture_change(tmp_path):
    from sd_aio import config, spade

    cfg = OmegaConf.load(config.DEFAULTS_PATH)
    saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved.model.pop("backend", None)
    OmegaConf.save(saved, tmp_path / "config.yaml")
    spade.validate_resume(cfg, tmp_path)
    cfg.model.backend = "flux"
    with pytest.raises(ValueError, match="model"):
        spade.validate_resume(cfg, tmp_path)
