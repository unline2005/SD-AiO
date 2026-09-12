import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Dinov2Config, Dinov2Model

from sd_aio import classifier
from sd_aio.spade import MultiScaleExtractor, SpadeConditionModule, _attach_unet_lora, _mark_lora_trainable
from sd_aio.vae_encoder import PreRestoreEncoder
from tests.helpers import (
    make_tiny_restorer,
    make_tiny_unet,
    make_tiny_vae,
)


def _tiny_classifier():
    encoder_config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=64,
        patch_size=8,
        num_channels=3,
        intermediate_size=64,
    )
    model = classifier.DegradationClassifier(num_classes=3, dino_path=None, freeze_encoder=True)
    model.encoder = Dinov2Model(encoder_config).requires_grad_(False)
    model.feature_dim = 16
    model.head = classifier.ClassifierHead(16, 3)
    return model


def test_classifier_protocol_loss_and_metrics():
    from omegaconf import OmegaConf

    model = _tiny_classifier()
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cfg = OmegaConf.create({"loss": {"focal_gamma": 2.0}})
    loss, logs = classifier.compute_loss(model, model, {"lq": images, "label": labels}, cfg)
    assert loss.shape == ()
    assert 0.0 < logs["loss"] < 3.0
    loss.backward()

    classifier.set_train_mode(model)
    assert model.head.training
    assert not model.encoder.training

    result = classifier.eval_step(model, model, {"lq": images, "label": labels, "task_name": ["a", "b"]})
    assert result["predictions"].shape == (2, 3)
    assert result["labels"].shape == (2, 3)


def test_deg_feature_extractor_has_differentiable_embedding_only():
    model = _tiny_classifier()
    extractor = classifier.DegFeatureExtractor(model, num_classes=3, inner_dim=16)
    extractor.set_trainable_embedding(True)
    features = extractor(torch.randn(2, 3, 64, 64))
    assert features.shape == (2, 16)
    features.sum().backward()
    assert extractor.deg_embedding.grad is not None
    assert extractor.deg_alpha.grad is not None
    assert model.encoder.embeddings.patch_embeddings.projection.weight.grad is None


def test_vae_encoder_adaln_channels_and_latent_mean_path():
    vae = make_tiny_vae()
    model = PreRestoreEncoder(
        vae.encoder,
        vae.config.block_out_channels,
        cond_dim=16,
        adaln_layers=["down2", "down3", "mid"],
    )
    model.frozen_vae = vae
    model.deg_extractor = torch.nn.Identity()
    latent = model(torch.randn(2, 3, 64, 64), torch.randn(2, 16))
    assert latent.shape == (2, 8, 8, 8)  # 2 * latent_channels
    z_mean = vae.quant_conv(latent)[:, :4]
    reconstructed = vae.decode(z_mean).sample
    assert reconstructed.shape == (2, 3, 64, 64)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert sum(parameter.numel() for parameter in model.adaln.parameters() if parameter.requires_grad) > 0


def test_spade_extractor_scales_match_latent_schedule():
    extractor = MultiScaleExtractor("simple-conv", channel_dims=(32, 64, 128))
    features = extractor(torch.randn(1, 3, 64, 64))
    assert features["C320"].shape == (1, 32, 8, 8)
    assert features["C640"].shape == (1, 64, 4, 4)
    assert features["C1280_Down"].shape == (1, 128, 2, 2)
    assert features["C1280_Mid"].shape == (1, 128, 1, 1)


def test_spade_injection_wraps_every_resnet_conv2_and_runs():
    unet = make_tiny_unet()
    condition = SpadeConditionModule("simple-conv", channel_dims=(32, 64, 128))
    condition.setup(unet)
    assert len(condition.wrappers) == 14
    condition.set_spatial_features(torch.randn(1, 3, 64, 64))
    output = unet(
        torch.randn(1, 4, 8, 8),
        torch.tensor([10]),
        encoder_hidden_states=torch.randn(1, 2, 16),
    ).sample
    assert output.shape == (1, 4, 8, 8)
    assert all(wrapper.current_cond_feat is None for wrapper in condition.wrappers)


def test_spade_restorer_single_forward_train_eval_shared():
    model = make_tiny_restorer("deg-aware")
    lq = torch.randn(1, 3, 64, 64)
    text = model.text_embedding_for(["Task"])
    prediction = model(lq, text, timestep=10)
    assert prediction.shape == (1, 3, 64, 64)
    assert torch.isfinite(prediction).all()

    batch = {"lq": lq, "gt": lq, "task_name": ["Task"], "image_id": ["test-image"]}
    from omegaconf import OmegaConf

    import sd_aio.spade as stage

    cfg = OmegaConf.create(
        {
            "loss": {
                "lambda_l2": 1.0,
                "lambda_lpips": 0.0,
                "timestep": {"strategy": "fixed", "value": 10},
            }
        }
    )
    loss, logs = stage.compute_loss(model, model, batch, cfg)
    assert "loss_l2" in logs
    loss.backward()
    result = stage.eval_step(model, model, batch)
    assert result["pred"].shape == (1, 3, 64, 64)


@pytest.mark.skipif(dist.is_initialized(), reason="process group already active")
def test_stage_loss_accepts_ddp_prepared_model_and_raw_model():
    model = make_tiny_restorer("simple")
    raw_model = model
    init_file = Path(tempfile.mkdtemp()) / "dist_init"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        ddp_model = DDP(model)
        lq = torch.randn(1, 3, 64, 64)
        batch = {"lq": lq, "gt": lq, "task_name": ["Task"], "image_id": ["test-image"]}
        cfg = OmegaConf.create(
            {"loss": {"lambda_l2": 1.0, "lambda_lpips": 0.0, "timestep": {"strategy": "fixed", "value": 10}}}
        )
        import sd_aio.spade as stage

        loss, logs = stage.compute_loss(ddp_model, raw_model, batch, cfg)
        assert "loss_l2" in logs
        loss.backward()
        assert sum(1 for parameter in raw_model.parameters() if parameter.grad is not None) > 0
    finally:
        dist.destroy_process_group()


def test_spade_wraps_lora_wrapped_conv2_without_breaking_forward():
    unet = make_tiny_unet()
    unet.requires_grad_(False)
    _attach_unet_lora(unet, rank=2, strategy="full")
    _mark_lora_trainable(unet)
    condition = SpadeConditionModule("simple-conv", channel_dims=(32, 64, 128))
    condition.setup(unet)
    condition.set_spatial_features(torch.randn(1, 3, 64, 64))
    output = unet(
        torch.randn(1, 4, 8, 8),
        torch.tensor([10]),
        encoder_hidden_states=torch.randn(1, 2, 16),
    ).sample
    assert output.shape == (1, 4, 8, 8)
    assert any("lora" in name for name, _ in unet.named_parameters())
