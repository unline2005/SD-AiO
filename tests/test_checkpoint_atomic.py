import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sd_aio import checkpoint


def make_model():
    return nn.Linear(2, 2)


def test_shape_error_does_not_partially_change_the_model(tmp_path):
    model = make_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / checkpoint.WEIGHTS_NAME
    save_file({"bias": torch.ones(2), "weight": torch.ones(3, 2)}, path)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        checkpoint.load_model_weights(model, path)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_latest_and_retention_skip_incomplete_writes(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    good = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    partial = tmp_path / "checkpoints" / "checkpoint-00000002"
    partial.mkdir()
    (partial / checkpoint.WRITING_NAME).touch()
    checkpoint.save_model_weights(model, partial / checkpoint.WEIGHTS_NAME)
    assert checkpoint.find_latest_checkpoint(tmp_path) == good
    latest = checkpoint.save_checkpoint(model, optimizer, None, 3, tmp_path, keep_last=1)
    assert partial.exists()
    assert not good.exists()
    assert checkpoint.find_latest_checkpoint(tmp_path) == latest


def test_existing_legacy_checkpoint_without_manifest_remains_usable(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    directory = checkpoint.save_checkpoint(model, optimizer, None, 5000, tmp_path)
    (directory / checkpoint.COMPLETE_NAME).unlink()
    assert checkpoint.find_latest_checkpoint(tmp_path) == directory
    step, restored = checkpoint.restore_training(model, optimizer, None, tmp_path)
    assert step == 5000
    assert restored == directory


def test_auxiliary_file_is_required_by_completion_manifest(tmp_path):
    class WithCondition(nn.Linear):
        def save_auxiliary(self, path):
            save_file({"condition": torch.ones(2)}, path.with_name("weights.condition.safetensors"))

    model = WithCondition(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    directory = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    content = json.loads((directory / checkpoint.COMPLETE_NAME).read_text())
    assert "weights.condition.safetensors" in content["files"]
    (directory / "weights.condition.safetensors").unlink()
    assert checkpoint.find_latest_checkpoint(tmp_path) is None
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint.resolve_weights_path(directory)


def test_training_roundtrip_keeps_optimizer_scheduler_and_weights(tmp_path):
    source = make_model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    source(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    directory = checkpoint.save_checkpoint(source, optimizer, scheduler, 1, tmp_path)
    target = make_model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.9)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
    step, _ = checkpoint.restore_training(target, target_optimizer, target_scheduler, tmp_path)
    assert step == 1
    assert torch.equal(source.weight, target.weight)
    assert scheduler.state_dict() == target_scheduler.state_dict()
    assert optimizer.param_groups[0]["lr"] == target_optimizer.param_groups[0]["lr"]
    assert checkpoint.is_checkpoint_complete(directory)


def test_failed_write_never_replaces_latest_or_deletes_prior_checkpoint(tmp_path, monkeypatch):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    good = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        checkpoint.save_checkpoint(model, optimizer, None, 2, tmp_path, keep_last=1)
    assert checkpoint.find_latest_checkpoint(tmp_path) == good


def test_final_directory_can_be_passed_directly(tmp_path):
    final = checkpoint.save_final(make_model(), tmp_path)
    assert checkpoint.resolve_weights_path(final) == final / checkpoint.WEIGHTS_NAME


def test_optimizer_step_mismatch_is_rejected(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    directory = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    (directory / checkpoint.COMPLETE_NAME).unlink()
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": None, "step": 2},
        directory / checkpoint.OPTIMIZER_NAME,
    )
    with pytest.warns(UserWarning, match="disagrees"), pytest.raises(RuntimeError, match="All checkpoints"):
        checkpoint.restore_training(model, optimizer, None, directory)


def test_failed_save_can_retry_same_step(tmp_path, monkeypatch):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    original = torch.save

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        checkpoint.save_checkpoint(model, optimizer, None, 2, tmp_path)
    monkeypatch.setattr(torch, "save", original)
    directory = checkpoint.save_checkpoint(model, optimizer, None, 2, tmp_path)
    assert checkpoint.find_latest_checkpoint(tmp_path) == directory
    assert checkpoint.is_checkpoint_complete(directory)


def test_repeated_save_of_same_step_replaces_complete_state(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    directory = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    with torch.no_grad():
        model.weight.fill_(0.25)
    checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    restored = make_model()
    checkpoint.load_model_weights(restored, directory / checkpoint.WEIGHTS_NAME)
    assert torch.equal(restored.weight, model.weight)
    assert list(directory.parent.iterdir()) == [directory]


def test_hidden_staging_directory_is_never_selected(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    directory = checkpoint.save_checkpoint(model, optimizer, None, 1, tmp_path)
    (directory.parent / ".checkpoint-99999999-abandoned").mkdir()
    assert checkpoint.find_latest_checkpoint(tmp_path) == directory


def test_explicit_file_in_partial_checkpoint_is_rejected(tmp_path):
    model = make_model()
    directory = tmp_path / "checkpoint-00000001"
    path = checkpoint.save_model_weights(model, directory / checkpoint.WEIGHTS_NAME)
    (directory / checkpoint.WRITING_NAME).touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint.resolve_weights_path(path)


class ConditionModel(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.register_buffer("condition", torch.randn(2))

    def save_auxiliary(self, path):
        save_file({"condition": self.condition}, path.with_name(path.stem + ".condition.safetensors"))

    def load_auxiliary(self, path):
        from safetensors.torch import load_file

        state = load_file(path.with_name(path.stem + ".condition.safetensors"))
        self.condition.copy_(state["condition"])


@pytest.mark.parametrize("final", [False, True])
def test_ema_saves_and_restores_its_frozen_condition(tmp_path, final):
    source = ConditionModel()
    ema = checkpoint.ModelEMA(source)
    if final:
        directory = checkpoint.save_final(source, tmp_path, ema)
    else:
        optimizer = torch.optim.AdamW(source.parameters())
        directory = checkpoint.save_checkpoint(source, optimizer, None, 1, tmp_path, ema)
    assert (directory / "ema.condition.safetensors").is_file()
    restored = ConditionModel()
    checkpoint.load_model_weights(restored, directory / checkpoint.EMA_NAME)
    assert torch.equal(restored.condition, source.condition)


def test_legacy_ema_can_use_existing_weights_condition(tmp_path):
    source = ConditionModel()
    directory = checkpoint.save_final(source, tmp_path, checkpoint.ModelEMA(source))
    (directory / "ema.condition.safetensors").unlink()
    restored = ConditionModel()
    checkpoint.load_model_weights(restored, directory / checkpoint.EMA_NAME)
    assert torch.equal(restored.condition, source.condition)


def test_failed_same_step_publication_restores_previous_checkpoint(tmp_path, monkeypatch):
    from pathlib import Path

    source = make_model()
    optimizer = torch.optim.AdamW(source.parameters())
    directory = checkpoint.save_checkpoint(source, optimizer, None, 1, tmp_path)
    expected = source.weight.detach().clone()
    with torch.no_grad():
        source.weight.fill_(0.25)
    original = Path.replace

    def fail_publication(self, target):
        if self.is_dir() and self.name.startswith(".checkpoint-") and "-old-" not in self.name:
            raise OSError("publication failed")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="publication failed"):
        checkpoint.save_checkpoint(source, optimizer, None, 1, tmp_path)
    restored = make_model()
    checkpoint.load_model_weights(restored, directory / checkpoint.WEIGHTS_NAME)
    assert torch.equal(restored.weight, expected)
    assert checkpoint.is_checkpoint_complete(directory)
