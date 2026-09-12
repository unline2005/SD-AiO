from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from sd_aio import checkpoint, classifier, vae_encoder
from tests.test_vae_pixel import make_cfg


def query_cfg(tmp_path):
    cfg = make_cfg(tmp_path, freeze=False, target="gt")
    cfg.model.condition_mode = "query_evidence"
    cfg.model.cond_dim = 32
    cfg.model.classifier = dict(
        head_type="query", query_heads=2, train_last_blocks=1, head_warmup_steps=0, l2sp_weight=0.0
    )
    trained = classifier.build_model(OmegaConf.merge(cfg, {"model": {"freeze_encoder": False}}))
    path = tmp_path / "query.safetensors"
    checkpoint.save_model_weights(trained, path)
    cfg.model.degradation_classifier_path = str(path)
    return cfg


def test_gated_evidence_matches_formula_and_is_frozen(tmp_path):
    cfg = query_cfg(tmp_path)
    extractor = classifier.build_deg_extractor(cfg)
    x = torch.rand(2, 3, 64, 64, requires_grad=True) * 2 - 1
    h, logits = extractor.classifier.forward_evidence(x)
    old_interface = extractor.classifier(x)
    torch.testing.assert_close(logits, old_interface, rtol=0, atol=0)
    condition = extractor(x)
    torch.testing.assert_close(condition, (logits.sigmoid().unsqueeze(-1) * h).flatten(1))
    assert condition.shape == (2, 32) and not condition.requires_grad
    assert not any(p.requires_grad for p in extractor.parameters())
    assert not hasattr(extractor, "deg_embedding")
    with torch.no_grad():
        extractor.classifier.head.bias[1] = -1000
    gated = extractor(x).reshape(2, 4, 8)
    assert torch.count_nonzero(gated[:, 1]) == 0


def test_query_shape_and_checkpoint_requirements_fail_fast(tmp_path):
    cfg = query_cfg(tmp_path)
    cfg.model.cond_dim = 768
    with pytest.raises(ValueError, match="cond_dim"):
        classifier.build_deg_extractor(cfg)
    cfg.model.cond_dim = 32
    path = Path(cfg.model.degradation_classifier_path)
    state = load_file(path)
    key = next(k for k in state if k.startswith("encoder.encoder.layer."))
    del state[key]
    save_file(state, path)
    with pytest.raises(RuntimeError, match="missing"):
        classifier.build_deg_extractor(cfg)


def test_vae_query_gradient_roundtrip_and_classifier_identity(tmp_path):
    cfg = query_cfg(tmp_path)
    model = vae_encoder.build_model(cfg)
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    batch = dict(lq=x, gt=torch.zeros_like(x))
    loss, _ = vae_encoder.compute_loss(model, model, batch, cfg)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaln.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.deg_extractor.parameters())
    assert all(p.grad is None for p in model.frozen_vae.parameters())
    optimizer = vae_encoder.make_optimizer(model, cfg)
    optimizer.step()
    folder = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path / "out")
    sidecar = load_file(folder / "weights.condition.safetensors")
    assert set(sidecar) == {"classifier_sha256"}
    torch.manual_seed(909)
    rebuilt = vae_encoder.build_model(cfg)
    checkpoint.restore_training(rebuilt, vae_encoder.make_optimizer(rebuilt, cfg), None, folder)
    with torch.no_grad():
        a, _ = vae_encoder._predict(model, model, x)
        b, _ = vae_encoder._predict(rebuilt, rebuilt, x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    sidecar["classifier_sha256"][0] ^= 1
    save_file(sidecar, folder / "weights.condition.safetensors")
    with pytest.raises(RuntimeError, match="classifier checkpoint differs"):
        checkpoint.load_model_weights(rebuilt, folder / "weights.safetensors")


def test_query_condition_two_ranks_and_resume(tmp_path):
    import json

    from tests.test_classifier_end_to_end import _launch_classifier

    cfg = query_cfg(tmp_path)
    cfg.loss.pixel_type = "l1"
    cfg.loss.lambda_lpips = 1.0
    cfg.eval.compute_lpips = True
    path = tmp_path / "query_vae.yaml"
    OmegaConf.save(cfg, path)
    log = _launch_classifier(path, tmp_path / "first.log")
    assert "Training finished at step 2" in log
    out = Path(cfg.output_dir)
    report = json.loads((out / "eval/metrics_step_00000001.json").read_text())
    assert len(report["vis_paths"]) == 4 and "lpips" in report["overall"]
    log = _launch_classifier(
        path, tmp_path / "resume.log", ("trainer.max_steps=3", "trainer.resume_from=latest")
    )
    assert "at step 2" in log and "Training finished at step 3" in log
    state = torch.load(out / "checkpoints/checkpoint-00000003/optimizer.pt", weights_only=True)
    assert state["step"] == state["scheduler"]["last_epoch"] == 3
    assert set(load_file(out / "final/weights.condition.safetensors")) == {"classifier_sha256"}
