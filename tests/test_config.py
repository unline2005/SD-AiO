import pytest

from sd_aio import config


def test_load_config_merges_defaults_and_tasks_file():
    cfg = config.load_config("configs/stage3_spade.yaml")
    assert cfg.stage == "spade"
    assert cfg.trainer.train_batch_size == 3
    assert cfg.trainer.gradient_accumulation_steps == 16
    assert cfg.data.num_workers == 8
    assert len(cfg.data.train) == 8
    assert cfg.data.test[1].name == "Test_Derain"
    assert cfg.data.test[1].lq_path.endswith("Rain100L/rainy")


def test_dot_path_overrides_are_typed():
    cfg = config.load_config(
        "configs/stage3_spade.yaml",
        ["trainer.max_steps=123", "model.lora.unet_rank=8", "loss.lambda_lpips=0"],
    )
    assert cfg.trainer.max_steps == 123
    assert cfg.model.lora.unet_rank == 8
    assert cfg.loss.lambda_lpips == 0


def test_required_key_missing_raises():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"a": {"b": None}})
    assert config.required(cfg, "a.b") is None
    with pytest.raises(KeyError):
        config.required(cfg, "a.missing")


def test_unknown_top_level_key_warns():
    from omegaconf import OmegaConf

    with pytest.warns(UserWarning):
        config.warn_unknown_keys(OmegaConf.create({"tyop_key": 1}))


def test_snapshot_roundtrip(tmp_path):
    cfg = config.load_config("configs/stage1_classifier.yaml")
    path = config.snapshot(cfg, tmp_path)
    assert path.exists()
    reloaded = config.load_config(path)
    assert reloaded.stage == cfg.stage


def test_override_without_equals_raises():
    with pytest.raises(ValueError):
        config.apply_overrides(None, ["bad_override"])


def test_classifier_snapshot_does_not_reload_external_tasks(tmp_path):
    from omegaconf import OmegaConf

    tasks = tmp_path / "tasks.yaml"
    OmegaConf.save(
        OmegaConf.create({"train": [{"name": "original", "deg_type": ["haze"]}], "test": []}),
        tasks,
    )
    experiment = tmp_path / "experiment.yaml"
    OmegaConf.save(
        OmegaConf.create({"stage": "classifier", "data": {"tasks_file": str(tasks)}}),
        experiment,
    )
    cfg = config.load_config(experiment)
    saved = config.snapshot(cfg, tmp_path / "run")
    tasks.unlink()
    reloaded = config.load_config(saved)
    assert "tasks_file" not in reloaded.data
    assert OmegaConf.to_container(reloaded.data.train) == OmegaConf.to_container(cfg.data.train)
    assert reloaded.data.test == cfg.data.test
