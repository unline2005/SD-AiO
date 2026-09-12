from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from transformers import Dinov2Config, Dinov2Model

from sd_aio import checkpoint, classifier


@pytest.fixture
def dino_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny-dinov2"
    encoder = Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            mlp_ratio=2,
        )
    )
    encoder.save_pretrained(path)
    return path


def make_classifier(path: Path, *, freeze: bool = True, hidden_dim: int = 8):
    return classifier.DegradationClassifier(
        num_classes=4,
        dino_path=str(path),
        freeze_encoder=freeze,
        head_hidden_dim=hidden_dim,
    )


def test_every_degradation_logit_receives_focal_supervision():
    logits = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    loss = classifier.focal_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(logits.grad.abs() > 0)
    assert torch.equal(logits.grad.sign(), torch.tensor([[-1.0, 1.0, -1.0, 1.0]]))
    assert torch.allclose(
        classifier.focal_loss(logits.detach(), labels, gamma=0.0),
        F.binary_cross_entropy_with_logits(logits.detach(), labels),
    )
    with pytest.raises(ValueError, match="matching"):
        classifier.focal_loss(torch.zeros(1, 4, 2), labels)


def test_training_eval_and_deg_features_use_independent_probabilities(dino_path):
    model = make_classifier(dino_path)
    classifier.set_eval_mode(model)
    image = torch.zeros(1, 3, 28, 28)
    with torch.no_grad():
        model.head.mlp[-1].weight.zero_()
        model.head.mlp[-1].bias.copy_(torch.tensor([2.0, 1.0, -1.0, -2.0]))
    batch = {"lq": image, "label": torch.tensor([[1.0, 1.0, 0.0, 0.0]])}
    cfg = OmegaConf.create({"loss": {"focal_gamma": 2.0}})
    _, logs = classifier.compute_loss(model, model, batch, cfg)
    result = classifier.eval_step(model, model, batch)
    assert logs["accuracy"] == 1.0
    assert torch.equal(result["predictions"], batch["label"].long())

    extractor = classifier.DegFeatureExtractor(model, num_classes=4)
    with torch.no_grad():
        extractor.deg_embedding.zero_()
        extractor.deg_embedding[:, :4].copy_(torch.eye(4))
        extractor.deg_alpha.fill_(1.0)
        cls_token, logits = model.forward_features(image)
        features = extractor(image)
    assert logits.shape == (1, 4)
    assert torch.allclose(
        (features - cls_token)[:, :4],
        torch.tensor([[0.8807971, 0.7310586, 0.2689414, 0.1192029]]),
        atol=1e-6,
    )
    assert torch.allclose(features[:, 4:], cls_token[:, 4:])


def test_dino_receives_imagenet_normalized_rgb(dino_path):
    model = make_classifier(dino_path)
    captured = []

    def capture(module, args, kwargs):
        captured.append(kwargs["pixel_values"].detach().clone())

    hook = model.encoder.register_forward_pre_hook(capture, with_kwargs=True)
    levels = torch.tensor([-1.0, 0.0, 1.0]).view(3, 1, 1, 1)
    with torch.no_grad():
        model(levels.expand(3, 3, 28, 28))
    hook.remove()
    expected = torch.tensor(
        [
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            [0.015 / 0.229, 0.044 / 0.224, 0.094 / 0.225],
            [0.515 / 0.229, 0.544 / 0.224, 0.594 / 0.225],
        ]
    ).view(3, 3, 1, 1)
    assert torch.allclose(captured[0], expected.expand_as(captured[0]), atol=1e-6)


@pytest.mark.parametrize("freeze", [True, False])
def test_encoder_gradient_and_mode_follow_freezing(dino_path, freeze):
    model = make_classifier(dino_path, freeze=freeze)
    classifier.set_eval_mode(model)
    classifier.set_train_mode(model)
    assert model.training and model.head.training
    assert model.encoder.training is (not freeze)
    assert all(module.training is (not freeze) for module in model.encoder.modules())
    images = torch.rand(1, 3, 28, 28) * 2.0 - 1.0
    logits = model(images)
    classifier.focal_loss(logits, torch.tensor([[1.0, 0.0, 1.0, 0.0]])).backward()
    assert torch.all(model.head.mlp[-1].bias.grad.abs() > 0)
    projection = model.encoder.embeddings.patch_embeddings.projection.weight
    if freeze:
        assert all(parameter.grad is None for parameter in model.encoder.parameters())
    else:
        assert projection.grad is not None
        assert torch.isfinite(projection.grad).all()
        assert projection.grad.abs().sum() > 0
        missing_grads = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing_grads == []


def test_finetuned_classifier_checkpoint_round_trip_and_old_head_rejection(dino_path, tmp_path):
    source = make_classifier(dino_path, freeze=False)
    classifier.set_eval_mode(source)
    with torch.no_grad():
        source.encoder.embeddings.cls_token.add_(0.5)
    path = checkpoint.save_model_weights(source, tmp_path / "classifier.safetensors")
    state = load_file(path)
    assert any(name.startswith("encoder.") for name in state)
    assert state["head.mlp.2.weight"].shape == (4, 8)
    target = make_classifier(dino_path, freeze=False)
    checkpoint.load_model_weights(target, path)
    classifier.set_eval_mode(target)
    images = torch.rand(1, 3, 28, 28) * 2.0 - 1.0
    with torch.no_grad():
        assert torch.allclose(source(images), target(images))

    old_state = {name: value.clone() for name, value in state.items()}
    old_state["head.mlp.2.weight"] = torch.zeros(8, 8)
    old_state["head.mlp.2.bias"] = torch.zeros(8)
    old_path = tmp_path / "old-two-logit-head.safetensors"
    save_file(old_state, old_path)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    with pytest.raises(RuntimeError, match=r"shape mismatch for head\.mlp\.2\."):
        checkpoint.load_model_weights(target, old_path)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_deg_extractor_loads_finetuned_encoder_and_custom_head(dino_path, tmp_path):
    source = make_classifier(dino_path, freeze=False, hidden_dim=12)
    with torch.no_grad():
        source.encoder.embeddings.cls_token.fill_(0.75)
    path = checkpoint.save_model_weights(source, tmp_path / "finetuned.safetensors")
    cfg = OmegaConf.create(
        {
            "model": {
                "num_deg_types": 4,
                "dino_path": str(dino_path),
                "head_hidden_dim": 12,
                "degradation_classifier_path": str(path),
                "cond_dim": 16,
                "train_deg_embedding": False,
            }
        }
    )
    extractor = classifier.build_deg_extractor(cfg)
    assert extractor.classifier.head.mlp[0].out_features == 12
    assert torch.equal(
        extractor.classifier.encoder.embeddings.cls_token,
        source.encoder.embeddings.cls_token,
    )
    assert not any(parameter.requires_grad for parameter in extractor.classifier.parameters())
