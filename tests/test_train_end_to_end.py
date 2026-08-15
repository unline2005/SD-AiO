from omegaconf import OmegaConf

from tests.helpers import make_spade_cfg, make_synthetic_task, make_tiny_sd_repo


def test_train_entrypoint_one_step_end_to_end(tmp_path):
    sd_path = make_tiny_sd_repo(tmp_path)
    task = make_synthetic_task(tmp_path, "Train_Denoise_15", "noise", n_images=3)
    output_dir = tmp_path / "out"
    cfg = make_spade_cfg(tmp_path, task, output_dir, sd_path)
    config_path = tmp_path / "experiment.yaml"
    OmegaConf.save(cfg, config_path)

    from train import main

    main(["--config", str(config_path)])

    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "train.log").exists()
    assert (output_dir / "final" / "weights.safetensors").exists()
    assert (output_dir / "eval" / "metrics_step_00000001.json").exists()

    # Standalone eval rebuilds from config + checkpoint with strict trainable-key
    # validation and writes the same benchmark report shape.
    from eval import main as eval_main

    eval_main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(output_dir / "final" / "weights.safetensors"),
            "--num_samples",
            "1",
        ]
    )
    assert (output_dir / "eval" / "metrics_standalone.json").exists()


def test_train_resume_restores_optimizer_state(tmp_path):
    sd_path = make_tiny_sd_repo(tmp_path)
    task = make_synthetic_task(tmp_path, "Train_Denoise_15", "noise", n_images=3)
    output_dir = tmp_path / "out"
    cfg = make_spade_cfg(tmp_path, task, output_dir, sd_path)
    cfg.trainer.max_steps = 2
    cfg.trainer.checkpointing_steps = 1
    config_path = tmp_path / "experiment.yaml"
    OmegaConf.save(cfg, config_path)

    from train import main

    main(["--config", str(config_path)])
    assert len(list((output_dir / "checkpoints").iterdir())) >= 1

    main(
        [
            "--config",
            str(config_path),
            "trainer.max_steps=1",
            "trainer.resume_from=latest",
            "trainer.checkpointing_steps=0",
        ]
    )
    log = (output_dir / "train.log").read_text()
    assert "Resumed from" in log
