"""Latent conventions and prediction math shared by diffusion backends.

Only sd_unet is an end-to-end restoration backend today. Flow prediction
and FLUX packing are tested building blocks, not SD3/FLUX model adapters.
The formulas follow diffusers DDPMScheduler, FlowMatchEulerDiscreteScheduler,
and FluxPipeline; sigma always means the scheduler sigma, never a timestep.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

DIFFUSION_PREDICTIONS = ("epsilon", "v_prediction", "sample")


@dataclass(frozen=True)
class LatentAffine:
    """VAE posterior -> backbone: (z - shift) * scale; decode uses the inverse."""

    scale: float
    shift: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.shift):
            raise ValueError("VAE latent scale must be finite and positive; shift must be finite")

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.shift) * self.scale if self.shift else latent * self.scale

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent / self.scale + self.shift if self.shift else latent / self.scale


def _coefficient(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value
    if value.ndim != 1 or value.numel() not in (1, reference.shape[0]):
        raise ValueError("Coefficient must be scalar or contain one value per batch item")
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def diffusion_x0(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    alpha_cumprod: torch.Tensor | float,
    prediction_type: str,
) -> torch.Tensor:
    """Recover x0 for x_t=sqrt(alpha)*x0+sqrt(1-alpha)*noise, without clipping."""
    if prediction.shape != noisy.shape:
        raise ValueError("Prediction and noisy latent shapes must match")
    alpha = _coefficient(alpha_cumprod, noisy)
    if prediction_type == "epsilon":
        return (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
    if prediction_type == "sample":
        return prediction
    raise ValueError(f"Unknown diffusion prediction {prediction_type!r}; use {DIFFUSION_PREDICTIONS}")


def flow_x0(noisy: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """For x_sigma=(1-sigma)*x0+sigma*noise and velocity=noise-x0; not diffusion v."""
    if velocity.shape != noisy.shape:
        raise ValueError("Velocity and noisy latent shapes must match")
    return noisy - _coefficient(sigma, noisy) * velocity


def pack_flux_latents(latent: torch.Tensor) -> torch.Tensor:
    """B,C,H,W -> B,(H/2)*(W/2),4*C in diffusers FLUX 2x2 patch order."""
    if latent.ndim != 4 or min(latent.shape) <= 0:
        raise ValueError("FLUX packing requires nonempty BCHW latents")
    batch, channels, height, width = latent.shape
    if height % 2 or width % 2:
        raise ValueError("FLUX latent height and width must be even")
    return (
        latent.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // 4, channels * 4)
    )


def unpack_flux_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse packing; height/width are latent dimensions, not RGB dimensions."""
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("FLUX latent height and width must be positive and even")
    if tokens.ndim != 3 or min(tokens.shape) <= 0:
        raise ValueError("FLUX unpacking requires nonempty BNC tokens")
    batch, count, channels = tokens.shape
    if count != height * width // 4 or channels % 4:
        raise ValueError("FLUX token shape does not match latent dimensions and 2x2 packing")
    return (
        tokens.reshape(batch, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, height, width)
    )


def require_backend(name: str) -> None:
    if name != "sd_unet":
        raise NotImplementedError(
            f"Backend {name!r} has no restoration adapter. Available: sd_unet. "
            "SD3/FLUX require transformer conditioning, text/pooled embeddings and flow scheduling."
        )


def validate_sd_unet_configs(
    unet: Mapping[str, Any],
    vae: Mapping[str, Any],
    text_encoder: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    *,
    condition_channels: Sequence[int] | None,
    text_dim: int | None,
) -> None:
    """Validate components before loading weights; support SD1/2 and SD-Turbo shapes."""
    if unet.get("_class_name") != "UNet2DConditionModel":
        raise ValueError("sd_unet requires UNet2DConditionModel, not a diffusion transformer")
    if vae.get("_class_name") != "AutoencoderKL":
        raise ValueError("sd_unet requires AutoencoderKL")
    if text_encoder.get("model_type") != "clip_text_model":
        raise ValueError("sd_unet requires a single CLIPTextModel")
    if str(scheduler.get("_class_name", "")).startswith("FlowMatch"):
        raise ValueError("Flow schedulers cannot be converted to a DDPM restoration schedule")
    if scheduler.get("prediction_type", "epsilon") not in DIFFUSION_PREDICTIONS:
        raise ValueError("Unsupported diffusion prediction_type")
    latent_channels = int(vae["latent_channels"])
    if unet["in_channels"] != latent_channels or unet["out_channels"] != latent_channels:
        raise ValueError("UNet input/output channels must match VAE latent_channels")
    if len(vae["block_out_channels"]) != 4:
        raise ValueError("sd_unet restoration requires a VAE with spatial compression factor 8")
    if vae.get("use_quant_conv") is False:
        raise ValueError("The pretrained encoder restoration path requires VAE quant_conv")
    if vae.get("latents_mean") is not None or vae.get("latents_std") is not None:
        raise ValueError("Per-channel latent normalization requires a dedicated backend adapter")
    if any(
        unet.get(key) is not None
        for key in ("addition_embed_type", "class_embed_type", "encoder_hid_dim_type", "time_cond_proj_dim")
    ):
        raise ValueError("Additional UNet conditioning is unsupported; SDXL/LCM need dedicated adapters")
    cross_dim = unet["cross_attention_dim"]
    if cross_dim != text_encoder["hidden_size"]:
        raise ValueError("UNet cross_attention_dim must match CLIP hidden_size")
    if text_dim is not None and text_dim != cross_dim:
        raise ValueError("model.text_dim must match UNet cross_attention_dim")
    if condition_channels is not None:
        channels = tuple(int(v) for v in condition_channels)
        if len(channels) != 3 or tuple(unet["block_out_channels"]) != (*channels, channels[-1]):
            raise ValueError("SPADE requires four UNet scales matching model.condition_channels")
        if any(c % 32 for c in channels):
            raise ValueError("SPADE condition channels must be divisible by 32")
