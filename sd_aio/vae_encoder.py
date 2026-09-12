"""Conditional VAE encoder training with latent or pixel supervision."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch import nn

from sd_aio import checkpoint, metrics, optim, utils
from sd_aio.classifier import QueryEvidenceCondition, build_deg_extractor
from sd_aio.config import DEFAULTS_PATH, resume_section
from sd_aio.losses import pixel_loss

_DEFAULTS = OmegaConf.load(DEFAULTS_PATH)


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
        factors = self.proj(f_deg).unsqueeze(-1).unsqueeze(-1)
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
        adaln_layers = list(adaln_layers)
        if len(block_out_channels) != len(self.encoder.down_blocks):
            raise ValueError(
                f"block_out_channels ({len(block_out_channels)}) != "
                f"encoder down_blocks ({len(self.encoder.down_blocks)})"
            )

        channels = {f"down{i}": int(block_out_channels[i]) for i in range(len(block_out_channels))}
        channels["mid"] = int(block_out_channels[-1])

        for name in adaln_layers:
            if name not in channels:
                raise ValueError(f"Unknown adaln layer {name}; expected one of {sorted(channels)}")
            self.adaln[name] = AdaIn(cond_dim, channels[name])

    def forward(self, pixel_values: torch.Tensor, f_deg: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(dtype=next(self.encoder.parameters()).dtype)
        f_deg = f_deg.to(dtype=next(self.adaln.parameters()).dtype)
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

    def save_auxiliary(self, weights_path):
        if hasattr(self, "deg_extractor"):
            save_condition(self.deg_extractor, weights_path)

    def load_auxiliary(self, weights_path):
        if hasattr(self, "deg_extractor"):
            load_condition(self.deg_extractor, weights_path)


def _condition_path(weights_path):
    path = Path(weights_path)
    return path.with_name(path.stem + ".condition.safetensors")


def save_condition(extractor, weights_path):
    path = _condition_path(weights_path)
    names = (
        ("classifier_sha256",)
        if isinstance(extractor, QueryEvidenceCondition)
        else ("deg_embedding", "deg_alpha")
    )
    state = {name: getattr(extractor, name).detach().cpu().contiguous() for name in names}
    temporary = path.with_suffix(".tmp")
    save_file(state, temporary)
    temporary.replace(path)


def load_condition(extractor, weights_path):
    path = _condition_path(weights_path)
    state = load_file(path)
    if isinstance(extractor, QueryEvidenceCondition):
        if set(state) != {"classifier_sha256"} or not torch.equal(
            state["classifier_sha256"], extractor.classifier_sha256.cpu()
        ):
            raise RuntimeError("Query condition classifier checkpoint differs from the saved experiment")
        return
    if set(state) != {"deg_embedding", "deg_alpha"}:
        raise RuntimeError(f"Invalid condition state: {path}")
    with torch.no_grad():
        for name, value in state.items():
            target = getattr(extractor, name)
            if target.shape != value.shape:
                raise RuntimeError(f"Condition shape mismatch for {name}")
            target.copy_(value.to(target))


def _options(cfg):
    return OmegaConf.merge(_DEFAULTS.model.vae_training, OmegaConf.select(cfg, "model.vae_training") or {})


def _load_pretrained_encoder(model: PreRestoreEncoder, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".safetensors":
        checkpoint.load_model_weights(model, path)
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
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
    options = _options(cfg)
    vae = AutoencoderKL.from_pretrained(str(model_cfg.sd_path), subfolder="vae")
    vae.requires_grad_(False).eval()

    deg_extractor = build_deg_extractor(cfg)
    deg_extractor.requires_grad_(False).eval()

    initialization = str(options.encoder_init)
    if initialization not in {"pretrained", "random"}:
        raise ValueError("encoder_init must be pretrained or random")
    if initialization == "random" and (options.freeze_encoder or model_cfg.get("pretrained_encoder_path")):
        raise ValueError("Random encoder must be trainable and cannot load an encoder checkpoint")
    encoder = vae.encoder
    if initialization == "random":
        encoder = AutoencoderKL.from_config(vae.config).encoder
    model = PreRestoreEncoder(
        encoder=encoder,
        block_out_channels=vae.config.block_out_channels,
        cond_dim=int(model_cfg.cond_dim),
        adaln_layers=list(model_cfg.adaln_layers),
    )
    model.encoder.requires_grad_(not bool(options.freeze_encoder))
    model.target_mode = str(options.target)
    if model.target_mode not in {"latent", "gt", "vae_reconstruction"}:
        raise ValueError(f"Unknown VAE target: {model.target_mode}")
    if model_cfg.train_deg_embedding:
        raise ValueError("Stage 2 keeps the classifier and condition embedding frozen")
    model.frozen_vae = vae
    model.deg_extractor = deg_extractor
    loss_options = OmegaConf.merge(_DEFAULTS.loss, cfg.loss)
    if str(loss_options.pixel_type) not in {"l1", "mse", "charbonnier"}:
        raise ValueError("loss.pixel_type must be l1, mse or charbonnier")
    lpips_weight = float(loss_options.lambda_lpips)
    if not math.isfinite(lpips_weight) or lpips_weight < 0:
        raise ValueError("loss.lambda_lpips must be finite and nonnegative")
    if lpips_weight > 0 and model.target_mode == "latent":
        raise ValueError("LPIPS requires a pixel-space VAE target")
    model.lpips = None
    if lpips_weight > 0:
        # Loading the fixed loss network must not change data sampling RNG state.
        with torch.random.fork_rng(devices=[]):
            model.lpips = metrics.load_lpips(str(loss_options.lpips_net))

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
    return optim.build_optimizer(trainable, cfg.optimizer)


def set_train_mode(model: PreRestoreEncoder) -> None:
    utils.set_train_mode(model)


def set_eval_mode(model: PreRestoreEncoder) -> None:
    utils.set_eval_mode(model)


def _latent_mean(model: PreRestoreEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    frozen_vae = model.frozen_vae
    vae_dtype = next(frozen_vae.parameters()).dtype
    latent = frozen_vae.quant_conv(frozen_vae.encoder(pixel_values.to(dtype=vae_dtype)))
    return latent.chunk(2, dim=1)[0]


def _predict(model, raw_model, lq):
    f_deg = raw_model.deg_extractor(lq)
    moments = raw_model.frozen_vae.quant_conv(model(lq, f_deg))
    mean = moments.chunk(2, dim=1)[0]
    return raw_model.frozen_vae.decode(mean).sample, mean


def compute_loss(model, raw_model, batch, cfg):
    precision = str(cfg.mixed_precision)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(batch["lq"].device.type, dtype=dtype, enabled=precision != "no"):
        if raw_model.target_mode == "latent":
            f_deg = raw_model.deg_extractor(batch["lq"])
            mean = raw_model.frozen_vae.quant_conv(model(batch["lq"], f_deg)).chunk(2, 1)[0]
            with torch.no_grad():
                target = _latent_mean(raw_model, batch["gt"])
            loss = F.l1_loss(mean.float(), target.float()) * float(cfg.loss.lambda_l1)
            return loss, {"loss": float(loss.detach()), "loss_l1": float(loss.detach())}
        with torch.no_grad():
            target = batch["gt"]
            if raw_model.target_mode == "vae_reconstruction":
                target = raw_model.frozen_vae.decode(_latent_mean(raw_model, target)).sample
        prediction, _ = _predict(model, raw_model, batch["lq"])
        loss_options = OmegaConf.merge(_DEFAULTS.loss, cfg.loss)
        pixel_type = str(loss_options.pixel_type)
        loss_pixel = pixel_loss(
            prediction, target, pixel_type, epsilon=float(loss_options.charbonnier_epsilon)
        )
    lpips_weight = float(loss_options.lambda_lpips)
    loss_lpips = loss_pixel.new_zeros(())
    if lpips_weight > 0:
        if raw_model.lpips is None:
            raise RuntimeError("LPIPS is enabled but its frozen network was not built")
        with torch.autocast(prediction.device.type, enabled=False):
            loss_lpips = raw_model.lpips(prediction.float(), target.float()).mean()
    loss = float(cfg.loss.lambda_pixel) * loss_pixel + lpips_weight * loss_lpips
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite VAE pixel loss")
    return loss, {
        "loss": float(loss.detach()),
        f"loss_pixel_{pixel_type}": float(loss_pixel.detach()),
        "loss_lpips": float(loss_lpips.detach()),
    }


def eval_step(model, raw_model, batch):
    prediction, _ = _predict(model, raw_model, batch["lq"])
    baseline = raw_model.frozen_vae.decode(_latent_mean(raw_model, batch["lq"])).sample
    target = raw_model.frozen_vae.decode(_latent_mean(raw_model, batch["gt"])).sample
    return {
        "pred": prediction.clamp(-1, 1),
        "gt": batch["gt"],
        "task_name": batch["task_name"],
        "baseline": baseline.clamp(-1, 1),
        "vae_gt": target.clamp(-1, 1),
    }


def validate_resume(cfg, resume_path):
    path = Path(resume_path)
    output = path.parent.parent if path.name.startswith("checkpoint-") else path
    saved = OmegaConf.load(output / "config.yaml")
    # Old pure-MSE snapshots predate the optional zero-weight LPIPS settings.
    saved.model = OmegaConf.merge(_DEFAULTS.model, saved.model)
    saved.loss = OmegaConf.merge(_DEFAULTS.loss, saved.loss)
    requested = OmegaConf.merge(
        cfg,
        {
            "model": OmegaConf.merge(_DEFAULTS.model, cfg.model),
            "loss": OmegaConf.merge(_DEFAULTS.loss, cfg.loss),
        },
    )
    for key in ("model", "data", "loss"):
        if resume_section(saved, key) != resume_section(requested, key):
            raise ValueError(f"VAE resume changes {key}; use the original experiment configuration")
