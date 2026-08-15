"""Shared helpers for fast CPU smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from sd_aio.spade import DegAwareConditionModule, SpadeRestorer


def make_synthetic_task(
    tmp_path: Path,
    name: str = "Train_Test",
    deg_type: str = "noise",
    n_images: int = 4,
    image_size: int = 64,
) -> dict:
    lq_dir = tmp_path / f"{name}_lq"
    gt_dir = tmp_path / f"{name}_gt"
    if deg_type == "noise":
        gt_dir = lq_dir
    lq_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_images):
        lq = Image.new("RGB", (image_size, image_size), (20 * index + 10, 90, 120))
        lq.save(lq_dir / f"{index:04d}.png")
        if deg_type != "noise":
            Image.new("RGB", (image_size, image_size), (20 * index + 10, 150, 160)).save(
                gt_dir / f"{index:04d}.png"
            )
    return {
        "name": name,
        "deg_type": deg_type,
        "lq_path": str(lq_dir),
        "gt_path": str(gt_dir),
        "prompt": "a b",
        "repeat_ratio": 1,
    }


def make_tiny_vae() -> AutoencoderKL:
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 16, 16, 16),
        layers_per_block=1,
        latent_channels=4,
        sample_size=64,
        norm_num_groups=4,
        scaling_factor=0.18215,
    )
    vae.requires_grad_(False).eval()
    return vae


def make_tiny_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=32,
    )


class FakeDegExtractor(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], self.dim)


def make_tiny_restorer(condition_type: str = "simple") -> SpadeRestorer:
    vae = make_tiny_vae()
    unet = make_tiny_unet()
    unet.requires_grad_(False)
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_start=1e-4, beta_end=0.02)

    condition = None
    deg_extractor = None
    if condition_type in ("simple", "deg-aware"):
        condition = DegAwareConditionModule(
            backbone_type="simple-conv",
            inner_dim=16,
            text_dim=16,
            channel_dims=(32, 64, 128),
        )
        condition.setup(unet)
    if condition_type == "deg-aware":
        deg_extractor = FakeDegExtractor(16)

    model = SpadeRestorer(
        vae=vae,
        unet=unet,
        noise_scheduler=scheduler,
        condition_module=condition,
        deg_extractor=deg_extractor,
        timestep=10,
    )
    model.prompt_embeddings["Task"] = nn.Parameter(torch.randn(1, 1, 16), requires_grad=False)
    return model


def make_tiny_sd_repo(tmp_path: Path) -> Path:
    """Create a tiny local SD 2.1-shaped repo for full entry-point tests."""
    root = tmp_path / "tiny-sd"
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    vocab: dict[str, int] = {}
    tokens = ["<|startoftext|>", "<|endoftext|>", "<|pad|>", *"abcdefghijklmnopqrstuvwxyz "]
    for token in tokens:
        vocab.setdefault(token, len(vocab))
    (tokenizer_dir / "vocab.json").write_text(json.dumps(vocab))
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n")
    CLIPTokenizer(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.txt")).save_pretrained(
        tokenizer_dir
    )

    text_config = CLIPTextConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=77,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    CLIPTextModel(text_config).save_pretrained(root / "text_encoder")
    make_tiny_vae().save_pretrained(root / "vae")
    make_tiny_unet().save_pretrained(root / "unet")
    DDPMScheduler(num_train_timesteps=50, beta_start=1e-4, beta_end=0.02).save_pretrained(root / "scheduler")
    return root


def make_spade_cfg(tmp_path: Path, task: dict, output_dir: Path, sd_path: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "stage": "spade",
            "seed": 0,
            "mixed_precision": "no",
            "pin_memory": False,
            "persistent_workers": False,
            "keep_last_checkpoints": 2,
            "output_dir": str(output_dir),
            "model": {
                "sd_path": str(sd_path),
                "condition_type": "simple",
                "backbone_type": "simple-conv",
                "cond_dim": 16,
                "num_deg_types": 3,
                "dino_path": None,
                "degradation_classifier_path": None,
                "pretrained_encoder_path": None,
                "timestep": 10,
                "train_deg_embedding": False,
                "condition_channels": [32, 64, 128],
                "text_dim": 16,
                "lora": {
                    "unet_rank": 0,
                    "vae_encoder_rank": 0,
                    "vae_decoder_rank": 0,
                    "strategy": "full",
                },
            },
            "data": {
                "train_image_size": 64,
                "num_workers": 0,
                "augmentation": {
                    "hflip_prob": 0.0,
                    "vflip_prob": 0.0,
                    "rot90_prob": 0.0,
                },
                "train": [task],
                "test": [task],
            },
            "optimizer": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "eps": 1e-8,
            },
            "scheduler": {"name": "cosine", "warmup_steps": 0},
            "loss": {
                "lambda_l2": 1.0,
                "lambda_lpips": 0.0,
                "timestep": {"strategy": "fixed", "value": 10},
            },
            "trainer": {
                "max_steps": 1,
                "train_batch_size": 1,
                "log_every": 1,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "eval_freq": 0,
                "checkpointing_steps": 0,
                "round_robin": False,
                "num_images_save_eval": 1,
                "eval_num_samples": None,
                "resume_from": None,
                "log_with": None,
            },
            "eval": {
                "batch_size": 1,
                "crop_to_multiple": 16,
                "pad_to_multiple": 64,
                "num_samples_per_task": None,
                "tiling": False,
                "tile_size": 32,
                "compute_lpips": False,
            },
            "ema": {"enabled": False, "decay": 0.999},
        }
    )
