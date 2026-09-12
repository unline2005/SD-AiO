"""Create a tiny, offline two-stage fixture; run it with train.py and eval.py."""

import sys
from pathlib import Path

import numpy as np
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image
from transformers import Dinov2Config, Dinov2Model


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/make_smoke_project.py NEW_OUTPUT_DIRECTORY")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    dino = root / "dino"
    Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=64,
            patch_size=14,
            mlp_ratio=2,
        )
    ).save_pretrained(dino)
    AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 16, 16, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
    ).save_pretrained(root / "sd" / "vae")
    rng = np.random.default_rng(42)
    tasks = {}
    for split in ("train", "val", "test"):
        for kind in ("lq", "gt"):
            (root / split / kind).mkdir(parents=True)
        for index in range(4):
            gt = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            lq = (gt.astype(np.float32) * 0.75 + 32).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt).save(root / split / "gt" / f"{index}.png")
            Image.fromarray(lq).save(root / split / "lq" / f"{index}.png")
        tasks[split] = [
            {
                "name": f"{split}_haze",
                "deg_type": "haze",
                "lq_path": str(root / split / "lq"),
                "gt_path": str(root / split / "gt"),
                "prompt": "a high quality clean image",
            }
        ]
    common = {
        "seed": 42,
        "mixed_precision": "no",
        "pin_memory": False,
        "persistent_workers": False,
        "data": {
            **tasks,
            "tasks_file": None,
            "num_workers": 0,
            "image_size": 64,
            "train_image_size": 64,
            "deg_types": ["haze", "rain", "snow", "lowlight"],
        },
        "model": {"dino_path": str(dino), "num_deg_types": 4, "head_hidden_dim": 8},
        "scheduler": {"name": "constant", "warmup_steps": 0},
        "trainer": {
            "max_steps": 2,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_split": "val",
            "eval_freq": 1,
            "eval_num_samples": 2,
            "checkpointing_steps": 1,
            "log_every": 1,
        },
        "eval": {
            "batch_size": 1,
            "num_samples_per_task": 2,
            "compute_lpips": False,
            "pad_to_multiple": 8,
            "crop_to_multiple": 8,
        },
    }
    classifier = OmegaConf.merge(
        common,
        {
            "stage": "classifier",
            "output_dir": str(root / "classifier_run"),
            "model": {"freeze_encoder": False},
            "optimizer": {"backbone_lr": 1e-4, "head_lr": 1e-3},
            "loss": {"focal_gamma": 2.0},
        },
    )
    vae = OmegaConf.merge(
        common,
        {
            "stage": "vae_encoder",
            "output_dir": str(root / "vae_run"),
            "model": {
                "sd_path": str(root / "sd"),
                "cond_dim": 16,
                "adaln_layers": ["down2", "down3", "mid"],
                "degradation_classifier_path": str(root / "classifier_run/final/weights.safetensors"),
                "pretrained_encoder_path": None,
                "train_deg_embedding": False,
                "vae_training": {"freeze_encoder": False, "target": "gt"},
            },
            "optimizer": {"lr": 1e-3},
            "loss": {"lambda_pixel": 1.0, "pixel_type": "l1"},
        },
    )
    OmegaConf.save(classifier, root / "classifier.yaml")
    OmegaConf.save(vae, root / "vae.yaml")
    print(f"Created {root}; run classifier.yaml, then vae.yaml through train.py")


if __name__ == "__main__":
    main()
