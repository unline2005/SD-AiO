import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from transformers import Dinov2Config, Dinov2Model

from sd_aio import checkpoint, classifier, config
from sd_aio.data import ClassificationDataset


def tiny_cfg(tmp_path):
    path = tmp_path / "dino"
    Dinov2Model(
        Dinov2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14)
    ).save_pretrained(path)
    return OmegaConf.merge(
        OmegaConf.load(config.DEFAULTS_PATH),
        {
            "model": {
                "dino_path": str(path),
                "num_deg_types": 4,
                "freeze_encoder": False,
                "head_hidden_dim": 8,
                "classifier": {
                    "head_type": "query",
                    "query_heads": 2,
                    "train_last_blocks": 1,
                    "head_warmup_steps": 1,
                    "l2sp_weight": 0.01,
                },
            },
            "optimizer": {"backbone_lr": 0.001, "head_lr": 0.01},
            "loss": {"classification": "asl"},
        },
    )


def test_query_independence_and_multilabel_gradients():
    head = classifier.LabelQueryHead(16, 4, 8, 2)
    tokens = torch.randn(2, 4, 16)
    before = head(tokens).detach()
    with torch.no_grad():
        head.queries[0].add_(torch.randn(8))
    after = head(tokens)
    torch.testing.assert_close(after[:, 1:], before[:, 1:])
    classifier.asymmetric_loss(after, torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 2), 0, 4, 0.05).backward()
    assert torch.all(head.weight.grad.abs().sum(-1) > 0)


def test_asl_bce_limit_and_extreme_logits():
    logits = torch.tensor([[-1000.0, 1000.0, 0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    loss = classifier.asymmetric_loss(logits, labels, 0, 0, 0)
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_partial_warmup_l2sp_and_resume(tmp_path):
    cfg = tiny_cfg(tmp_path)
    model = classifier.build_model(cfg)
    classifier.set_train_mode(model)
    assert not model.encoder.encoder.layer[0].training
    assert model.encoder.encoder.layer[1].training
    assert classifier.l2sp_loss(model).item() == 0
    initial = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
    optimizer = classifier.make_optimizer(model, cfg)
    batch = {"lq": torch.randn(2, 3, 28, 28), "label": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)}
    for step in range(2):
        loss, _ = classifier.compute_loss(model, model, batch, cfg)
        loss.backward()
        classifier.before_optimizer_step(model, cfg, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 0:
            for n, p in model.encoder.named_parameters():
                torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
    assert classifier.l2sp_loss(model).item() > 0
    assert any(not torch.equal(p, initial[n]) for n, p in model.encoder.named_parameters() if p.requires_grad)
    for n, p in model.encoder.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
    folder = checkpoint.save_checkpoint(model, optimizer, None, 2, tmp_path / "out")
    state = load_file(folder / "weights.safetensors")
    assert not any("_l2sp" in n or "encoder.encoder.layer.0." in n for n in state)
    rebuilt = classifier.build_model(cfg)
    restored_optimizer = classifier.make_optimizer(rebuilt, cfg)
    step, _ = checkpoint.restore_training(rebuilt, restored_optimizer, None, folder)
    assert step == 2
    torch.testing.assert_close(classifier.l2sp_loss(rebuilt), classifier.l2sp_loss(model))
    loss, _ = classifier.compute_loss(rebuilt, rebuilt, batch, cfg)
    loss.backward()
    classifier.before_optimizer_step(rebuilt, cfg, step)
    assert all(p.grad is not None for p in rebuilt.encoder.parameters() if p.requires_grad)


def test_clean_and_sampling_weights(tmp_path):
    Image.new("RGB", (28, 28)).save(tmp_path / "image.png")
    tasks = [
        dict(name="haze", deg_type="haze", lq_path=str(tmp_path), sampling_weight=2),
        dict(name="clean", clean=True, deg_type=[], lq_path=str(tmp_path), sampling_weight=1),
    ]
    dataset = ClassificationDataset(tasks, ["haze", "rain", "snow", "lowlight"], image_size=28, training=True)
    assert dataset[1]["label"].sum() == 0
    torch.testing.assert_close(dataset.task_sampling_weights(), torch.tensor([2.0, 1.0], dtype=torch.double))
    tasks[1]["sampling_weight"] = float("nan")
    with pytest.raises(ValueError, match="sampling_weight"):
        ClassificationDataset(tasks, ["haze"], image_size=28, training=True)


def test_single_protocol_configs():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tasks = OmegaConf.load(root / "configs/tasks_classifier_single.yaml")
    for split in ("train", "val"):
        assert all(len(t.deg_type) <= 1 for t in tasks[split])
        assert any(t.get("clean") for t in tasks[split])
    assert any(len(t.deg_type) > 1 for t in tasks.test)
