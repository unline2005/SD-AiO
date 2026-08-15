import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sd_aio import checkpoint


class CheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.trainable = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(self.frozen(x))


def test_save_load_trainable_only_weights(tmp_path):
    source = CheckpointModel()
    with torch.no_grad():
        source.trainable.weight.fill_(0.25)
    path = checkpoint.save_model_weights(source, tmp_path / "weights.safetensors")

    target = CheckpointModel()
    checkpoint.load_model_weights(target, path)
    assert torch.allclose(target.trainable.weight, source.trainable.weight)


def test_empty_checkpoint_raises(tmp_path):
    model = CheckpointModel()
    save_file({}, tmp_path / "empty.safetensors")
    with pytest.raises(RuntimeError, match="empty"):
        checkpoint.load_model_weights(model, tmp_path / "empty.safetensors")


def test_unexpected_checkpoint_key_raises(tmp_path):
    model = CheckpointModel()
    save_file({"trainable.extra": torch.zeros(1)}, tmp_path / "bad.safetensors")
    try:
        checkpoint.load_model_weights(model, tmp_path / "bad.safetensors")
    except RuntimeError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_missing_trainable_key_raises(tmp_path):
    model = CheckpointModel()
    save_file({"frozen.weight": torch.zeros(4, 4)}, tmp_path / "bad.safetensors")
    try:
        checkpoint.load_model_weights(model, tmp_path / "bad.safetensors")
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_checkpoint_layout_and_resume(tmp_path):
    model = CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    saved = checkpoint.save_checkpoint(model, optimizer, None, 10, tmp_path)
    assert (saved / checkpoint.WEIGHTS_NAME).exists()
    assert (saved / checkpoint.OPTIMIZER_NAME).exists()

    restored = CheckpointModel()
    step, directory = checkpoint.restore_training(restored, optimizer, None, tmp_path)
    assert step == 10
    assert directory == saved
    assert torch.allclose(restored.trainable.weight, model.trainable.weight)
    assert checkpoint.resolve_weights_path(tmp_path).name == checkpoint.WEIGHTS_NAME


def test_keep_last_checkpoints_prunes_old(tmp_path):
    model = CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (10, 20, 30):
        checkpoint.save_checkpoint(model, optimizer, None, step, tmp_path, keep_last=2)
    names = [path.name for path in checkpoint.iter_checkpoints(tmp_path)]
    assert names == ["checkpoint-00000030", "checkpoint-00000020"]


def test_restore_training_requires_optimizer_state(tmp_path):
    model = CheckpointModel()
    checkpoint_dir = tmp_path / "checkpoint-00000001"
    checkpoint_dir.mkdir()
    checkpoint.save_model_weights(model, checkpoint_dir / checkpoint.WEIGHTS_NAME)
    with (
        pytest.warns(UserWarning, match="optimizer.pt does not exist"),
        pytest.raises(RuntimeError, match="All checkpoints failed"),
    ):
        checkpoint.restore_training(model, torch.optim.AdamW(model.parameters()), None, checkpoint_dir)


def test_ema_tracks_trainable_parameters_only():
    model = CheckpointModel()
    ema = checkpoint.ModelEMA(model, decay=0.999)
    assert set(ema.shadow) == {"trainable.weight", "trainable.bias"}
    with torch.no_grad():
        model.trainable.weight.mul_(2.0)
    ema.update(model)
    assert not torch.allclose(ema.shadow["trainable.weight"], model.trainable.weight.detach())


def test_ema_load_checks_missing_and_unexpected(tmp_path):
    model = CheckpointModel()
    ema = checkpoint.ModelEMA(model)
    missing_file = tmp_path / "missing.safetensors"
    save_file({"trainable.weight": ema.shadow["trainable.weight"]}, missing_file)
    try:
        ema.load_state_dict(missing_file)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    unexpected_file = tmp_path / "unexpected.safetensors"
    state = ema.state_dict()
    state["extra.weight"] = torch.zeros(1)
    save_file(state, unexpected_file)
    try:
        ema.load_state_dict(unexpected_file)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_resolve_weights_path_ema_errors_are_explicit(tmp_path):
    model = CheckpointModel()
    checkpoint_dir = tmp_path / "checkpoint-00000001"
    checkpoint_dir.mkdir()
    checkpoint.save_model_weights(model, checkpoint_dir / checkpoint.WEIGHTS_NAME)

    with pytest.raises(FileNotFoundError, match="use_ema"):
        checkpoint.resolve_weights_path(checkpoint_dir, prefer_ema=True)
    with pytest.raises(ValueError, match="use_ema"):
        checkpoint.resolve_weights_path(checkpoint_dir / checkpoint.WEIGHTS_NAME, prefer_ema=True)


def test_resolve_weights_path_prefers_final_ema(tmp_path):
    model = CheckpointModel()
    ema = checkpoint.ModelEMA(model)
    checkpoint.save_final(model, tmp_path, ema=ema)
    assert checkpoint.resolve_weights_path(tmp_path, prefer_ema=True).name == checkpoint.EMA_NAME
    assert checkpoint.resolve_weights_path(tmp_path).name == checkpoint.WEIGHTS_NAME
