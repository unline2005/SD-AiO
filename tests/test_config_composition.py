from pathlib import Path

import pytest
from omegaconf import OmegaConf

from sd_aio import config


def write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(content), path)
    return path


def test_nested_bases_are_relative_and_later_base_wins(tmp_path):
    write_yaml(tmp_path / "base.yaml", {"stage": "vae_encoder", "optimizer": {"lr": 1e-4}})
    write_yaml(tmp_path / "nested" / "common.yaml", {"_base_": "../base.yaml", "seed": 11})
    write_yaml(tmp_path / "second.yaml", {"optimizer": {"lr": 2e-4}})
    path = write_yaml(
        tmp_path / "experiment.yaml",
        {
            "_base_": ["nested/common.yaml", "second.yaml"],
            "optimizer": {"weight_decay": 0.0},
        },
    )
    cfg = config.load_config(path, ["optimizer.lr=0.0003"])
    assert cfg.optimizer.lr == 3e-4
    assert cfg.optimizer.weight_decay == 0
    assert cfg.seed == 11
    assert "_base_" not in cfg


def test_circular_bases_fail_with_chain(tmp_path):
    write_yaml(tmp_path / "one.yaml", {"_base_": "two.yaml"})
    path = write_yaml(tmp_path / "two.yaml", {"_base_": "one.yaml"})
    with pytest.raises(ValueError, match=r"Circular config inheritance.*two.yaml"):
        config.load_config(path)


@pytest.mark.parametrize(
    "content, match",
    [
        ({"stage": "typo_stage"}, "Unknown stage"),
        ({"stage": "vae_encoder", "trainer": {"train_batch_size": 0}}, "train_batch_size"),
        ({"stage": "vae_encoder", "trainer": {"max_steps": 1.5}}, "max_steps"),
        ({"stage": "vae_encoder", "optimizer": {"lr": float("nan")}}, "optimizer.lr"),
        ({"stage": "vae_encoder", "mixed_precision": "typo"}, "mixed_precision"),
    ],
)
def test_invalid_config_fails_before_model_construction(tmp_path, content, match):
    with pytest.raises(ValueError, match=match):
        config.load_config(write_yaml(tmp_path / "experiment.yaml", content))


@pytest.mark.parametrize("stage", ["classifier", "vae_encoder", "spade"])
def test_snapshot_embeds_tasks_for_every_stage(tmp_path, stage):
    tasks = write_yaml(
        tmp_path / "tasks.yaml",
        {
            "train": [{"name": "original"}],
            "val": [],
            "test": [{"name": "test"}],
        },
    )
    experiment = write_yaml(
        tmp_path / "experiment.yaml",
        {
            "stage": stage,
            "data": {"tasks_file": str(tasks)},
        },
    )
    cfg = config.load_config(experiment, ["data.train.0.name=override"])
    assert cfg.data.train[0].name == "override"
    saved = config.snapshot(cfg, tmp_path / "run")
    tasks.unlink()
    actual = config.load_config(saved)
    assert "tasks_file" not in actual.data
    assert actual.data.train == cfg.data.train
    assert config.resume_section(actual, "data") == config.resume_section(cfg, "data")


def test_training_validation_does_not_break_inference_only_config(tmp_path):
    cfg = config.load_config(write_yaml(tmp_path / "inference.yaml", {"stage": "spade"}))
    with pytest.raises(KeyError, match="output_dir"):
        config.validate_training(cfg)


def test_resume_section_normalizes_new_defaults():
    old = OmegaConf.create({"model": {}, "loss": {}, "data": {"train": [], "tasks_file": "gone.yaml"}})
    new = OmegaConf.merge(OmegaConf.load(config.DEFAULTS_PATH), {"data": {"train": []}})
    for section in ("model", "loss", "data"):
        assert config.resume_section(old, section) == config.resume_section(new, section)


def test_hub_ids_and_existing_project_path_semantics_are_preserved(tmp_path):
    cfg = config.load_config(
        write_yaml(
            tmp_path / "inference.yaml",
            {
                "stage": "spade",
                "model": {"sd_path": "stabilityai/sd-turbo"},
                "output_dir": "./output",
            },
        )
    )
    assert cfg.model.sd_path == "stabilityai/sd-turbo"
    assert Path(cfg.output_dir) == config.PROJECT_ROOT / "output"


def test_manifest_path_resolved_but_manifest_root_stays_relative(tmp_path):
    cfg = config.load_config(
        write_yaml(
            tmp_path / "inference.yaml",
            {
                "stage": "spade",
                "data": {"test": [{"manifest": "./pairs.jsonl", "manifest_root": "../images"}]},
            },
        )
    )
    assert cfg.data.test[0].manifest == str(config.PROJECT_ROOT / "pairs.jsonl")
    assert cfg.data.test[0].manifest_root == "../images"


@pytest.mark.parametrize("value", ["image_equal", "source_equal", None])
def test_unsupported_metric_aggregation_is_rejected(tmp_path, value):
    path = write_yaml(tmp_path / "experiment.yaml", {"stage": "spade", "eval": {"overall": value}})
    with pytest.raises(ValueError, match=r"eval\.overall"):
        config.load_config(path)


def test_inference_uses_visible_metric_aggregation_default(tmp_path):
    path = write_yaml(tmp_path / "inference.yaml", {"stage": "spade"})
    cfg = config.load_config(path)
    assert cfg.eval.overall == OmegaConf.load(config.DEFAULTS_PATH).eval.overall == "task_equal"


def test_training_validation_merges_defaults_without_mutating_input(tmp_path):
    cfg = OmegaConf.create(
        {
            "stage": "spade",
            "output_dir": str(tmp_path),
            "optimizer": {"lr": 1e-4},
            "data": {"train": [{"name": "synthetic"}]},
            "trainer": {
                "train_batch_size": 1,
                "max_steps": 1,
                "log_every": 0,
                "eval_freq": 0,
                "checkpointing_steps": 0,
            },
        }
    )
    original = OmegaConf.to_container(cfg, resolve=False)
    config.validate_training(cfg)
    assert OmegaConf.to_container(cfg, resolve=False) == original
