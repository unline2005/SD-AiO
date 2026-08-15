"""Stage 2: VAE encoder pre-training (latent-mean alignment).

Trains a deep copy of the SD VAE encoder whose down/mid features are modulated
by AdaIN from the frozen degradation feature ``F_Deg``.  Loss: L1 between the
LQ and HQ latent means after ``quant_conv`` (``z[:, :4]``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from torch import nn

from sd_aio import checkpoint, utils
from sd_aio.classifier import build_deg_extractor


class AdaIn(nn.Module):
    """F_Deg -> MLP -> (gamma, beta) per-channel modulation."""

    def __init__(self, cond_dim: int, channel_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, channel_dim),
            nn.SiLU(),
            nn.Linear(channel_dim, channel_dim * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, features: torch.Tensor, f_deg: torch.Tensor) -> torch.Tensor:
        factors = self.proj(f_deg).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        gamma, beta = factors.chunk(2, dim=1)
        return features * (1.0 + gamma) + beta


class PreRestoreEncoder(nn.Module):
    """Deep-copied VAE encoder with AdaIN after the selected down/mid blocks."""

    def __init__(
        self,
        encoder: nn.Module,
        block_out_channels: tuple[int, ...] | list[int],
        cond_dim: int = 768,
        adaln_layers: tuple[str, ...] | list[str] = ("down2", "down3", "mid"),
    ) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.adaln = nn.ModuleDict()
        self._adaln_layers = list(adaln_layers)

        # diffusers Encoder: the i-th down block outputs block_out_channels[i].
        channels = {f"down{i}": int(block_out_channels[i]) for i in range(len(block_out_channels))}
        channels["mid"] = int(block_out_channels[-1])

        for name in self._adaln_layers:
            if name not in channels:
                raise ValueError(f"Unknown adaln layer {name}; expected one of {sorted(channels)}")
            self.adaln[name] = AdaIn(cond_dim, channels[name])

    def forward(self, pixel_values: torch.Tensor, f_deg: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder.conv_in(pixel_values)
        for index, down_block in enumerate(self.encoder.down_blocks):
            hidden = down_block(hidden)
            name = f"down{index}"
            if name in self.adaln:
                hidden = self.adaln[name](hidden, f_deg)
        hidden = self.encoder.mid_block(hidden)
        if "mid" in self.adaln:
            hidden = self.adaln["mid"](hidden, f_deg)
        hidden = self.encoder.conv_norm_out(hidden)
        hidden = self.encoder.conv_act(hidden)
        hidden = self.encoder.conv_out(hidden)
        return hidden


def _load_pretrained_encoder(model: PreRestoreEncoder, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        checkpoint.load_model_weights(model, path)
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder" in state:
        state = state["encoder"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Stage-2 checkpoint has unexpected keys: {unexpected[:5]}")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_missing = [name for name in missing if name in trainable_names]
    if trainable_missing:
        raise RuntimeError(f"Stage-2 checkpoint is missing trainable keys: {trainable_missing[:5]}")


def build_model(cfg: OmegaConf, device: torch.device | None = None) -> PreRestoreEncoder:
    model_cfg = cfg.model
    vae = AutoencoderKL.from_pretrained(str(model_cfg.sd_path), subfolder="vae")
    vae.requires_grad_(False).eval()

    deg_extractor = build_deg_extractor(cfg, device)
    deg_extractor.requires_grad_(False).eval()

    model = PreRestoreEncoder(
        encoder=vae.encoder,
        block_out_channels=vae.config.block_out_channels,
        cond_dim=int(model_cfg.cond_dim),
        adaln_layers=list(model_cfg.adaln_layers),
    )
    model.frozen_vae = vae
    model.deg_extractor = deg_extractor
    model.scaling_factor = float(vae.config.scaling_factor)

    checkpoint_path = model_cfg.get("pretrained_encoder_path")
    if checkpoint_path:
        _load_pretrained_encoder(model, checkpoint_path)

    if device is not None:
        model = model.to(device)
    return model


def make_optimizer(model: PreRestoreEncoder, cfg: OmegaConf) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Stage-2 model has no trainable parameters")
    optimizer_cfg = cfg.optimizer
    return torch.optim.AdamW(
        trainable,
        lr=float(optimizer_cfg.lr),
        betas=(float(optimizer_cfg.betas[0]), float(optimizer_cfg.betas[1])),
        weight_decay=float(optimizer_cfg.weight_decay),
        eps=float(optimizer_cfg.eps),
    )


def set_train_mode(model: PreRestoreEncoder) -> None:
    utils.set_train_mode(model)


def set_eval_mode(model: PreRestoreEncoder) -> None:
    utils.set_eval_mode(model)


def _latent_mean(model: PreRestoreEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    frozen_vae = model.frozen_vae
    vae_dtype = next(frozen_vae.parameters()).dtype
    latent = frozen_vae.quant_conv(frozen_vae.encoder(pixel_values.to(dtype=vae_dtype)))
    return latent[:, :4]


def compute_loss(
    model: PreRestoreEncoder,
    raw_model: PreRestoreEncoder,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> tuple[torch.Tensor, dict[str, float]]:
    lq = batch["lq"]
    hq = batch["gt"]
    f_deg = raw_model.deg_extractor(lq)
    z_lq_raw = model(lq, f_deg)
    z_lq = raw_model.frozen_vae.quant_conv(z_lq_raw.to(dtype=next(raw_model.frozen_vae.parameters()).dtype))[
        :, :4
    ]
    with torch.no_grad():
        z_hq = _latent_mean(raw_model, hq)
    loss = F.l1_loss(z_lq.float(), z_hq.float()) * float(cfg.loss.lambda_l1)
    return loss, {"loss": float(loss.detach()), "loss_l1": float(loss.detach())}


def eval_step(
    model: PreRestoreEncoder,
    raw_model: PreRestoreEncoder,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> dict[str, Any]:
    lq = batch["lq"]
    f_deg = raw_model.deg_extractor(lq)
    z_lq_raw = model(lq, f_deg)
    z_mean = raw_model.frozen_vae.quant_conv(
        z_lq_raw.to(dtype=next(raw_model.frozen_vae.parameters()).dtype)
    )[:, :4]
    prediction = raw_model.frozen_vae.decode(z_mean).sample.clamp(-1.0, 1.0)
    return {
        "pred": prediction,
        "gt": batch["gt"],
        "lq": lq,
        "task_name": batch["task_name"],
    }
