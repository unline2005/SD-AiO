import json

import pytest
import torch

from sd_aio.runtime import MetricWindow, append_metrics, write_metadata


def test_metric_window_averages_and_detaches():
    window = MetricWindow()
    window.add({"loss": torch.tensor(2.0, requires_grad=True), "accuracy": 0.5})
    window.add({"loss": torch.tensor(4.0, requires_grad=True), "accuracy": 1.0})
    values = window.pop()
    assert values["loss"] == 3.0 and not values["loss"].requires_grad
    assert values["accuracy"] == 0.75
    assert window.pop() == {}
    window.add({"loss": 8.0})
    assert window.pop()["loss"] == 8.0
    with pytest.raises(ValueError, match="scalar"):
        window.add({"loss": torch.ones(2)})


def test_runtime_metadata_and_jsonl(tmp_path):
    path = write_metadata(tmp_path, world_size=2, seed=42)
    payload = json.loads(path.read_text())
    assert payload["world_size"] == 2 and payload["seed"] == 42
    assert "train.py" in payload["code_sha256"]
    append_metrics(tmp_path, 3, {"loss": 0.5})
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == {"step": 3, "loss": 0.5}
