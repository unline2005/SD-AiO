"""Real two-process CPU integration test for classifier joint fine-tuning."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from transformers import Dinov2Config, Dinov2Model

from sd_aio import config as configlib


def _launch_classifier(config_path, log_path, overrides=()):
    project = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(project / "train.py"),
        "--config",
        str(config_path),
        *overrides,
    ]
    environment = os.environ.copy()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        ACCELERATE_USE_CPU="true",
        TOKENIZERS_PARALLELISM="false",
        TORCH_DISTRIBUTED_DEBUG="OFF",
        PYTHONFAULTHANDLER="1",
    )
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            pytest.fail(f"Classifier CPU DDP timed out; log: {log_path}\n{log_path.read_text()[-10000:]}")
    assert returncode == 0, log_path.read_text()[-16000:]
    return log_path.read_text()


@pytest.mark.parametrize("e1", [False, True])
def test_classifier_two_cpu_ranks_joint_training_eval_and_resume(tmp_path, e1):
    torch.manual_seed(19)
    dino_path = tmp_path / "tiny-dino"
    dino = Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            image_size=28,
            patch_size=14,
        )
    )
    dino.save_pretrained(dino_path)
    initial_encoder = {name: tensor.clone() for name, tensor in dino.state_dict().items()}
    labels = ["haze", "rain", "snow", "lowlight"]
    train_tasks = []
    test_tasks = []
    for index, degradations in enumerate([*labels, ["haze", "rain"]]):
        for split, tasks in (("train", train_tasks), ("test", test_tasks)):
            folder = tmp_path / "images" / split / str(index)
            folder.mkdir(parents=True)
            for image_index in range(2):
                image = Image.new("RGB", (40, 36), (25 * index + 10, 90, 40 * image_index + 20))
                image.save(folder / f"{image_index:03d}.png")
            tasks.append(
                {
                    "name": f"{split}_{index}",
                    "deg_type": degradations,
                    "lq_path": str(folder),
                    "repeat_ratio": 1,
                }
            )
    output = tmp_path / "training-output"
    cfg = OmegaConf.merge(
        OmegaConf.load(configlib.DEFAULTS_PATH),
        {
            "stage": "classifier",
            "output_dir": str(output),
            "seed": 9,
            "mixed_precision": "no",
            "pin_memory": False,
            "persistent_workers": False,
            "keep_last_checkpoints": 4,
            "model": {
                "dino_path": str(dino_path),
                "num_deg_types": 4,
                "freeze_encoder": False,
                "head_hidden_dim": 16,
            },
            "data": {
                "tasks_file": None,
                "deg_types": labels,
                "classification_sampling": "task_balanced",
                "image_size": 28,
                "num_workers": 0,
                "augmentation": {"hflip_prob": 0.0},
                "train": train_tasks,
                "test": test_tasks,
            },
            "optimizer": {"backbone_lr": 0.001, "head_lr": 0.005},
            "scheduler": {"name": "constant_with_warmup", "warmup_steps": 1},
            "loss": {"focal_gamma": 2.0},
            "trainer": {
                "max_steps": 2,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "log_every": 1,
                "eval_freq": 1,
                "eval_num_samples": 1,
                "checkpointing_steps": 1,
            },
            "eval": {"batch_size": 4, "compute_lpips": False},
        },
    )
    if e1:
        cfg.model.classifier = {
            "head_type": "query",
            "query_heads": 2,
            "train_last_blocks": 1,
            "head_warmup_steps": 1,
            "l2sp_weight": 0.01,
        }
        cfg.loss.classification = "asl"
    config_path = tmp_path / "classifier.yaml"
    OmegaConf.save(cfg, config_path)
    first_log = _launch_classifier(config_path, tmp_path / "two-rank-train.log")
    assert "Training finished at step 2" in first_log

    for step in (1, 2):
        saved = torch.load(
            output / "checkpoints" / f"checkpoint-{step:08d}" / "optimizer.pt", weights_only=True
        )
        assert saved["step"] == saved["scheduler"]["last_epoch"] == step
        assert all(
            int(state["step"]) in ({step, step - 1} if e1 else {step})
            for state in saved["optimizer"]["state"].values()
        )
    trained = load_file(output / "final" / "weights.safetensors")
    assert any(name.startswith("encoder.") for name in trained)
    assert any(name.startswith("head.") for name in trained)
    assert "encoder.embeddings.mask_token" not in trained
    assert any(
        not torch.equal(tensor, initial_encoder[name.removeprefix("encoder.")])
        for name, tensor in trained.items()
        if name.startswith("encoder.")
    )
    periodic = json.loads((output / "eval" / "metrics_step_00000001.json").read_text())
    assert periodic["classification"]["num_samples"] == 5
    assert periodic["classification"]["class_names"] == labels
    assert [entry["name"] for entry in periodic["classification"]["per_class"]] == labels

    second_log = _launch_classifier(
        config_path,
        tmp_path / "two-rank-resume.log",
        ("trainer.max_steps=3", "trainer.resume_from=latest"),
    )
    assert "at step 2" in second_log
    assert "Training finished at step 3" in second_log
    resumed = torch.load(output / "checkpoints" / "checkpoint-00000003" / "optimizer.pt", weights_only=True)
    assert resumed["step"] == resumed["scheduler"]["last_epoch"] == 3
    assert all(
        int(state["step"]) in ({2, 3} if e1 else {3}) for state in resumed["optimizer"]["state"].values()
    )
    final_report = json.loads((output / "eval" / "metrics_step_00000003.json").read_text())
    assert final_report["classification"]["class_names"] == labels
    assert final_report["classification"]["num_samples"] == 10


def test_classifier_raw_cpu_bf16_eval_with_float32_weights(tmp_path):
    from collections import OrderedDict

    from sd_aio import classifier
    from sd_aio.eval import run_eval

    dino_path = tmp_path / "tiny-dino-bf16-eval"
    Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=28,
            patch_size=14,
        )
    ).save_pretrained(dino_path)
    names = ["haze", "rain", "snow", "lowlight"]
    cfg = OmegaConf.create(
        {
            "stage": "classifier",
            "mixed_precision": "bf16",
            "model": {
                "dino_path": str(dino_path),
                "num_deg_types": 4,
                "freeze_encoder": False,
                "head_hidden_dim": 16,
            },
            "data": {"deg_types": names},
            "eval": {"num_samples_per_task": None},
        }
    )
    model = classifier.build_model(cfg, device=torch.device("cpu"))
    batch = {
        "lq": torch.zeros(2, 3, 28, 28),
        "label": torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]]),
    }
    report = run_eval(
        classifier,
        model,
        model,
        OrderedDict([("composite", [batch])]),
        cfg,
        device=torch.device("cpu"),
        save_images=False,
    )
    assert report.classification["num_samples"] == 2
    assert report.classification["class_names"] == names
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.isfinite(torch.tensor(value)) for value in report.overall.values())
