"""CPU regressions for the classifier branch of the sole training loop."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from accelerate.optimizer import AcceleratedOptimizer
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader


class TinyClassifierStage:
    def __init__(self):
        self.loss_calls = 0
        self.fail_at = None

    def build_model(self, cfg, device):
        return torch.nn.Linear(2, 4).to(device)

    def make_optimizer(self, model, cfg):
        return torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.0)

    def set_train_mode(self, model):
        model.train()

    def compute_loss(self, model, raw_model, batch, cfg):
        self.loss_calls += 1
        if self.loss_calls == self.fail_at:
            raise RuntimeError("intentional interruption")
        logits = model(batch["lq"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"].float())
        return loss, {"loss": float(loss.detach())}


def _configure_training(monkeypatch, tmp_path, accumulation, world_size):
    import train

    cfg = OmegaConf.create(
        {
            "stage": "classifier",
            "output_dir": str(tmp_path / "output"),
            "seed": 7,
            "mixed_precision": "no",
            "data": {
                "deg_types": ["haze", "rain", "snow", "lowlight"],
                "train": [{"name": "synthetic", "deg_type": "haze"}],
            },
            "optimizer": {"head_lr": 0.2, "backbone_lr": 0.2},
            "scheduler": {"name": "linear", "warmup_steps": 1},
            "ema": {"enabled": False},
            "eval": {"compute_lpips": False},
            "keep_last_checkpoints": 8,
            "trainer": {
                "distributed_timeout_seconds": 600,
                "train_batch_size": 1,
                "gradient_accumulation_steps": accumulation,
                "max_steps": 4,
                "max_grad_norm": 1.0,
                "log_every": 0,
                "eval_freq": 0,
                "checkpointing_steps": 1,
                "resume_from": None,
            },
        }
    )
    sample = {"lq": torch.tensor([0.1, -0.2]), "label": torch.tensor([1, 0, 0, 0])}
    loader = DataLoader([sample] * 64, batch_size=1)
    stage = TinyClassifierStage()
    monkeypatch.setattr(train.configlib, "load_config", lambda *args: cfg)
    monkeypatch.setattr(train.datalib, "build_loaders", lambda *args, **kwargs: (loader, OrderedDict()))
    monkeypatch.setattr(train.utils, "load_stage", lambda name: stage)
    # Exercise AcceleratedScheduler's real world-size branch without launching GPUs/DDP.
    monkeypatch.setattr(
        "accelerate.scheduler.AcceleratorState", lambda: SimpleNamespace(num_processes=world_size)
    )
    return train, cfg, stage


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("skip_update", [False, True])
def test_classifier_scheduler_counts_successful_optimizer_updates(
    monkeypatch, tmp_path, world_size, accumulation, skip_update
):
    train, _cfg, stage = _configure_training(monkeypatch, tmp_path, accumulation, world_size)
    original_step = AcceleratedOptimizer.step
    skipped = False

    def step_with_overflow(self, *args, **kwargs):
        nonlocal skipped
        if self.gradient_state.sync_gradients:
            if skip_update and not skipped:
                self._is_overflow = True
                skipped = True
                return None
            self._is_overflow = False
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(AcceleratedOptimizer, "step", step_with_overflow)
    train.main(["--config", "unused-test-config.yaml"])
    assert stage.loss_calls == accumulation * (4 + int(skip_update))
    for step in range(1, 5):
        saved = torch.load(
            tmp_path / "output" / "checkpoints" / f"checkpoint-{step:08d}" / "optimizer.pt",
            weights_only=True,
        )
        assert saved["step"] == step
        assert saved["scheduler"]["last_epoch"] == step
        assert saved["scheduler"]["_last_lr"][0] == pytest.approx(0.2 * (4 - step) / 3)
        assert all(int(value["step"]) == step for value in saved["optimizer"]["state"].values())


def test_classifier_resume_preserves_scheduler_and_optimizer_updates(monkeypatch, tmp_path):
    train, cfg, stage = _configure_training(monkeypatch, tmp_path, accumulation=2, world_size=2)
    uninterrupted = tmp_path / "uninterrupted"
    cfg.output_dir = str(uninterrupted)
    train.main(["--config", "unused-test-config.yaml"])

    resumed = tmp_path / "resumed"
    cfg.output_dir = str(resumed)
    stage.loss_calls = 0
    stage.fail_at = 5
    with pytest.raises(RuntimeError, match="intentional interruption"):
        train.main(["--config", "unused-test-config.yaml"])
    interrupted = torch.load(
        resumed / "checkpoints" / "checkpoint-00000002" / "optimizer.pt", weights_only=True
    )
    assert interrupted["step"] == interrupted["scheduler"]["last_epoch"] == 2

    stage.fail_at = None
    cfg.trainer.resume_from = "latest"
    train.main(["--config", "unused-test-config.yaml"])
    expected = load_file(uninterrupted / "final" / "weights.safetensors")
    actual = load_file(resumed / "final" / "weights.safetensors")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    final = torch.load(resumed / "checkpoints" / "checkpoint-00000004" / "optimizer.pt", weights_only=True)
    assert final["step"] == final["scheduler"]["last_epoch"] == 4
