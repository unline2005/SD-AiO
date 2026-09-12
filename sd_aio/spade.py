"""Stage 3: single-step SD 2.1 restoration with pluggable SPADE conditioning.

``SpadeRestorer`` owns the entire forward path (encode -> add noise -> denoise
-> x0 estimate -> decode), so training and inference can never drift apart.
The optional condition module lives on the UNet ResBlock ``conv2`` layers.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.resnet import ResnetBlock2D
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch import nn
from torchvision import models as torchvision_models
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel

from sd_aio import config as configlib
from sd_aio import metrics, optim, utils
from sd_aio.backends import (
    DIFFUSION_PREDICTIONS,
    LatentAffine,
    diffusion_x0,
    require_backend,
    validate_sd_unet_configs,
)
from sd_aio.classifier import build_deg_extractor
from sd_aio.vae_encoder import (
    PreRestoreEncoder,
    _load_pretrained_encoder,
    load_condition,
    save_condition,
)

BackboneType = Literal["simple-conv", "resnet18", "convnext_tiny"]

UNET_LORA_TARGETS = {
    "only_attn": ["to_k", "to_q", "to_v", "to_out.0"],
    "only_mlp": [
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "conv_out",
        "proj_in",
        "proj_out",
        "ff.net.2",
        "ff.net.0.proj",
    ],
    "full": [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "conv_out",
        "proj_in",
        "proj_out",
        "ff.net.2",
        "ff.net.0.proj",
    ],
}
VAE_LORA_TARGET = r"\.(conv1|conv2|conv_in|conv_shortcut|conv_out|to_k|to_q|to_v|to_out\.0)$"


class Spade(nn.Module):
    """GroupNorm followed by per-pixel gamma/beta predicted from the condition feature."""

    def __init__(self, output_channels: int, cond_input_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=output_channels, affine=False)
        padding = kernel_size // 2
        self.shared = nn.Sequential(
            nn.Conv2d(cond_input_channels, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.gamma = nn.Conv2d(128, output_channels, kernel_size=kernel_size, padding=padding)
        self.beta = nn.Conv2d(128, output_channels, kernel_size=kernel_size, padding=padding)
        for layer in (self.gamma, self.beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        if cond_feat.shape[2:] != features.shape[2:]:
            raise ValueError(
                f"SPADE condition {tuple(cond_feat.shape)} does not match features {tuple(features.shape)}"
            )
        normalized = self.norm(features)
        shared = self.shared(cond_feat)
        return features + normalized * self.gamma(shared) + self.beta(shared)


class SpadeWrapper(nn.Module):
    def __init__(self, target_module: nn.Conv2d, condition_channels: int) -> None:
        super().__init__()
        self.target_module = target_module
        base = getattr(target_module, "base_layer", target_module)
        self.spade = Spade(base.out_channels, condition_channels)
        self.current_cond_feat: torch.Tensor | None = None

    def forward(self, input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.current_cond_feat is None:
            raise RuntimeError("SPADE condition feature was not set before the UNet forward")
        if hasattr(self.target_module, "base_layer"):
            hidden_states = self.target_module(input_tensor)
        else:
            hidden_states = self.target_module(input_tensor, *args, **kwargs)
        cond_feat = self.current_cond_feat.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = self.spade(hidden_states, cond_feat)
        self.current_cond_feat = None
        return hidden_states


class MultiScaleExtractor(nn.Module):
    """LQ image pyramid at UNet down/mid/up scales (latent 64/32/16/8)."""

    def __init__(
        self,
        backbone_type: BackboneType = "simple-conv",
        channel_dims: tuple[int, int, int] = (320, 640, 1280),
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.C320, self.C640, self.C1280 = (int(dim) for dim in channel_dims)
        if self.backbone_type == "simple-conv":
            self.conv_in = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.SiLU(),
                nn.Conv2d(128, self.C320, 4, 2, 1),
                nn.SiLU(),
            )
            self.down_C640 = nn.Sequential(nn.Conv2d(self.C320, self.C640, 4, 2, 1), nn.SiLU())
            self.down_C1280_D = nn.Sequential(nn.Conv2d(self.C640, self.C1280, 4, 2, 1), nn.SiLU())
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.C1280, self.C1280, 4, 2, 1), nn.SiLU())
        elif self.backbone_type == "resnet18":
            resnet = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer_320_raw, self.layer_640_raw, self.layer_1280_raw = (
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            self.proj_320 = nn.Conv2d(128, self.C320, 1)
            self.proj_640 = nn.Conv2d(256, self.C640, 1)
            self.proj_1280 = nn.Conv2d(512, self.C1280, 1)
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.C1280, self.C1280, 3, 2, 1), nn.SiLU())
        elif self.backbone_type == "convnext_tiny":
            features = torchvision_models.convnext_tiny(
                weights=torchvision_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            ).features
            self.stem = nn.Sequential(features[0], features[1])
            self.layer_320_raw = nn.Sequential(features[2], features[3])
            self.layer_640_raw = nn.Sequential(features[4], features[5])
            self.layer_1280_raw = nn.Sequential(features[6], features[7])
            self.proj_320 = nn.Conv2d(192, self.C320, 1)
            self.proj_640 = nn.Conv2d(384, self.C640, 1)
            self.proj_1280 = nn.Conv2d(768, self.C1280, 1)
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.C1280, self.C1280, 3, 2, 1), nn.GELU())
        else:
            raise ValueError(
                f"Unsupported backbone_type {backbone_type}; use simple-conv/resnet18/convnext_tiny"
            )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.backbone_type == "simple-conv":
            f_320 = self.conv_in(images)
            f_640 = self.down_C640(f_320)
            f_1280_d = self.down_C1280_D(f_640)
            f_1280_m = self.down_C1280_M(f_1280_d)
        else:
            hidden = self.stem(images)
            h_320 = self.layer_320_raw(hidden)
            h_640 = self.layer_640_raw(h_320)
            h_1280 = self.layer_1280_raw(h_640)
            f_320 = self.proj_320(h_320)
            f_640 = self.proj_640(h_640)
            f_1280_d = self.proj_1280(h_1280)
            f_1280_m = self.down_C1280_M(f_1280_d)
        return {
            "C320": f_320,
            "C640": f_640,
            "C1280_Down": f_1280_d,
            "C1280_Mid": f_1280_m,
        }


class SpadeConditionModule(nn.Module):
    """Spatial-only SPADE conditioning."""

    DOWN_KEYS: ClassVar[dict[int, str]] = {
        0: "C320",
        1: "C640",
        2: "C1280_Down",
        3: "C1280_Mid",
    }
    UP_KEYS: ClassVar[dict[int, str]] = {
        0: "C1280_Mid",
        1: "C1280_Down",
        2: "C640",
        3: "C320",
    }

    def __init__(
        self,
        backbone_type: BackboneType = "simple-conv",
        channel_dims: tuple[int, int, int] = (320, 640, 1280),
    ) -> None:
        super().__init__()
        self.extractor = MultiScaleExtractor(backbone_type, channel_dims=channel_dims)
        self._scale_groups: dict[str, list[SpadeWrapper]] = {key: [] for key in self.DOWN_KEYS.values()}
        self.wrappers = nn.ModuleList()

    def setup(self, unet: nn.Module) -> None:
        expected_channels = {
            "C320": self.extractor.C320,
            "C640": self.extractor.C640,
            "C1280_Down": self.extractor.C1280,
            "C1280_Mid": self.extractor.C1280,
        }

        def hook(resnet: ResnetBlock2D, key: str) -> None:
            conv2 = getattr(resnet, "conv2", None)
            if conv2 is None:
                return
            base = getattr(conv2, "base_layer", conv2)
            if base.out_channels != expected_channels[key]:
                raise ValueError(
                    f"UNet conv2 channels ({base.out_channels}) do not match "
                    f"condition_channels[{key}] ({expected_channels[key]})"
                )
            wrapper = SpadeWrapper(conv2, base.out_channels)
            self._scale_groups[key].append(wrapper)
            self.wrappers.append(wrapper)
            resnet.conv2 = wrapper

        for index, block in enumerate(unet.down_blocks):
            for resnet in block.resnets:
                hook(resnet, self.DOWN_KEYS[index])
        for resnet in getattr(unet.mid_block, "resnets", []):
            hook(resnet, "C1280_Mid")
        for index, block in enumerate(unet.up_blocks):
            for resnet in block.resnets:
                hook(resnet, self.UP_KEYS[index])

    def set_spatial_features(self, lq_image: torch.Tensor) -> None:
        features = self.extractor(lq_image)
        for key, wrappers in self._scale_groups.items():
            for wrapper in wrappers:
                wrapper.current_cond_feat = features[key]

    def forward(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        f_deg: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        self.set_spatial_features(lq_image)
        return text_embedding


class DegTextFusion(nn.Module):
    """F_Deg -> one token prepended to the CLIP text embedding."""

    def __init__(self, inner_dim: int = 768, text_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(inner_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))

    def forward(self, f_deg: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        token = self.proj(f_deg).unsqueeze(1)
        return torch.cat([token, text_embedding], dim=1)


class DegAwareConditionModule(SpadeConditionModule):
    """SPADE spatial modulation + degradation token for cross-attention."""

    def __init__(
        self,
        backbone_type: BackboneType = "simple-conv",
        inner_dim: int = 768,
        text_dim: int = 1024,
        channel_dims: tuple[int, int, int] = (320, 640, 1280),
    ) -> None:
        super().__init__(backbone_type, channel_dims=channel_dims)
        self.text_fusion = DegTextFusion(inner_dim=inner_dim, text_dim=text_dim)

    def forward(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        f_deg: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        text_embedding = super().forward(lq_image, text_embedding, f_deg)
        if f_deg is None:
            raise RuntimeError("deg-aware conditioning requires a degradation feature extractor")
        if text_embedding is not None:
            text_embedding = self.text_fusion(f_deg, text_embedding)
        return text_embedding


def _attach_unet_lora(unet: nn.Module, rank: int, strategy: str) -> None:
    if strategy not in UNET_LORA_TARGETS:
        raise ValueError(f"Unknown unet lora strategy {strategy}; use {sorted(UNET_LORA_TARGETS)}")
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=UNET_LORA_TARGETS[strategy],
    )
    unet.add_adapter(config, adapter_name="restoration")


def _attach_vae_lora(vae: nn.Module, encoder_rank: int = 0, decoder_rank: int = 0) -> None:
    def attach(rank: int, pattern: str, name: str) -> None:
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=pattern,
        )
        vae.add_adapter(config, adapter_name=name)

    if encoder_rank <= 0 and decoder_rank <= 0:
        return
    if encoder_rank == decoder_rank:
        parts = "|".join(
            part for part, rank in (("encoder", encoder_rank), ("decoder", decoder_rank)) if rank > 0
        )
        attach(encoder_rank, rf"^({parts}).*{VAE_LORA_TARGET}", "restoration")
        return
    if encoder_rank > 0:
        attach(encoder_rank, rf"^encoder.*{VAE_LORA_TARGET}", "restoration_encoder")
    if decoder_rank > 0:
        attach(decoder_rank, rf"^decoder.*{VAE_LORA_TARGET}", "restoration_decoder")


def _mark_lora_trainable(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = "lora" in name


def _sample_timestep(loss_cfg: OmegaConf) -> int:
    timestep_cfg = loss_cfg.timestep
    if timestep_cfg.get("strategy") == "range":
        low, high = timestep_cfg.range
        return int(torch.randint(int(low), int(high) + 1, (1,)).item())
    return int(timestep_cfg.value)


class SpadeRestorer(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        noise_scheduler: DDPMScheduler,
        condition_module: nn.Module | None = None,
        deg_extractor: nn.Module | None = None,
        pretrained_encoder: PreRestoreEncoder | None = None,
        lpips_model: nn.Module | None = None,
        timestep: int = 100,
        eval_noise_seed: int = 42,
        tokenizer: Any | None = None,
        text_encoder: CLIPTextModel | None = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.condition_module = condition_module
        self.deg_extractor = deg_extractor
        self.pretrained_encoder = pretrained_encoder
        self.lpips = lpips_model
        self.default_timestep = int(timestep)
        self.eval_noise_seed = int(eval_noise_seed)
        self.prediction_type = str(noise_scheduler.config.prediction_type)
        if self.prediction_type not in DIFFUSION_PREDICTIONS:
            raise ValueError(f"Unsupported diffusion prediction_type: {self.prediction_type}")
        self.latent_affine = LatentAffine(
            float(vae.config.scaling_factor), float(vae.config.shift_factor or 0.0)
        )
        self.scaling_factor = self.latent_affine.scale
        self.prompt_embeddings = nn.ParameterDict()

        self._aux: dict[str, Any] = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }

    def noise_seeds(self, image_ids):
        return [zlib.crc32(f"{self.eval_noise_seed}:{name}".encode()) for name in image_ids]

    @staticmethod
    def noise_like(reference, seeds):
        if seeds is None:
            return torch.randn_like(reference)
        if len(seeds) != reference.shape[0]:
            raise ValueError("One noise seed is required per image")
        return torch.stack(
            [
                torch.randn(
                    reference.shape[1:],
                    device=reference.device,
                    dtype=reference.dtype,
                    generator=torch.Generator(device=reference.device).manual_seed(seed),
                )
                for seed in seeds
            ]
        )

    def save_auxiliary(self, weights_path):
        if self.deg_extractor is not None:
            save_condition(self.deg_extractor, weights_path)

    def load_auxiliary(self, weights_path):
        if self.deg_extractor is not None:
            load_condition(self.deg_extractor, weights_path)

    def text_embedding_for(self, task_names: list[str]) -> torch.Tensor:
        missing = [name for name in task_names if name not in self.prompt_embeddings]
        if missing:
            raise KeyError(
                f"No cached prompt for task(s) {missing}; available: {list(self.prompt_embeddings)}"
            )
        return torch.cat([self.prompt_embeddings[name] for name in task_names], dim=0)

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        tokenizer = self._aux.get("tokenizer")
        text_encoder = self._aux.get("text_encoder")
        if tokenizer is None or text_encoder is None:
            raise RuntimeError("Tokenizer/text encoder are not available for custom prompts")
        tokens = tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(text_encoder.device)
        with torch.no_grad():
            embedding = text_encoder(tokens)[0].detach()
        return embedding.to(device=device if device is not None else embedding.device, dtype=dtype)

    def encode_lq(
        self, lq_image: torch.Tensor, f_deg: torch.Tensor | None = None, noise_seeds=None
    ) -> torch.Tensor:
        if self.pretrained_encoder is not None:
            if f_deg is None:
                f_deg = self.deg_extractor(lq_image) if self.deg_extractor is not None else None
            if f_deg is None:
                raise RuntimeError("pretrained_encoder requires a degradation feature extractor")
            z_raw = self.pretrained_encoder(lq_image, f_deg)
            z_mean = self.vae.quant_conv(z_raw)[:, : self.vae.config.latent_channels]
            return self.latent_affine.encode(z_mean)
        posterior = self.vae.encode(lq_image).latent_dist
        if noise_seeds is None:
            latent = posterior.sample()
        else:
            latent = posterior.mean + posterior.std * self.noise_like(
                posterior.mean, [seed ^ 0x5A5A5A5A for seed in noise_seeds]
            )
        return self.latent_affine.encode(latent)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(self.latent_affine.decode(latent)).sample

    def _x0_coeff(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[
            timesteps
        ]
        alphas = alphas.view(-1, 1, 1, 1)
        return ((1.0 - alphas) ** 0.5) / (alphas**0.5)

    def forward(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        timestep: int | None = None,
        noise_seeds: list[int] | None = None,
    ) -> torch.Tensor:
        lq_image = lq_image.to(dtype=next(self.vae.parameters()).dtype)
        f_deg = self.deg_extractor(lq_image) if self.deg_extractor is not None else None
        z0 = self.encode_lq(lq_image, f_deg=f_deg, noise_seeds=noise_seeds)
        noise = self.noise_like(z0, noise_seeds)
        if timestep is None:
            timestep = self.default_timestep
        timesteps = torch.full((z0.shape[0],), int(timestep), device=z0.device, dtype=torch.long)
        z_t = self.noise_scheduler.add_noise(z0, noise, timesteps)

        if self.condition_module is not None:
            text_embedding = self.condition_module(lq_image, text_embedding, f_deg=f_deg)
        if text_embedding is None:
            raise RuntimeError("SpadeRestorer.forward requires text_embedding")
        noise_pred = self.unet(z_t, timesteps, encoder_hidden_states=text_embedding).sample
        if self.prediction_type == "epsilon":
            coeff = self._x0_coeff(timesteps).to(device=z0.device, dtype=z0.dtype)
            denoised = z0 + coeff * (noise - noise_pred)
        else:
            alphas = self.noise_scheduler.alphas_cumprod.to(z_t.device)[timesteps]
            denoised = diffusion_x0(z_t, noise_pred, alphas, self.prediction_type)
        return self.decode_latent(denoised)


def _populate_prompt_embeddings(model: SpadeRestorer, tasks: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for task in tasks:
        name = str(task["name"])
        prompt = str(task.get("prompt", ""))
        if name in seen:
            if seen[name] != prompt:
                raise ValueError(f"Task {name} appears with two different prompts")
            continue
        seen[name] = prompt
        embedding = model.encode_prompt(prompt, device=None, dtype=torch.float32)
        model.prompt_embeddings[name] = nn.Parameter(embedding[0:1], requires_grad=False)


def _build_condition_module(cfg: OmegaConf, unet: nn.Module) -> nn.Module | None:
    condition_type = str(configlib.required(cfg, "model.condition_type"))
    if condition_type == "none":
        return None
    backbone = str(configlib.required(cfg, "model.backbone_type"))
    channel_dims = tuple(int(dim) for dim in cfg.model.get("condition_channels", [320, 640, 1280]))
    if condition_type == "simple":
        module = SpadeConditionModule(backbone_type=backbone, channel_dims=channel_dims)
    elif condition_type in ("deg-aware", "deg_aware_sft"):
        module = DegAwareConditionModule(
            backbone_type=backbone,
            inner_dim=int(configlib.required(cfg, "model.cond_dim")),
            text_dim=int(cfg.model.get("text_dim", 1024)),
            channel_dims=channel_dims,
        )
    else:
        raise ValueError(f"Unknown condition_type {condition_type}; use none/simple/deg-aware")
    module.setup(unet)
    return module


def build_model(cfg: OmegaConf, device: torch.device | None = None) -> SpadeRestorer:
    model_cfg = cfg.model
    if int(model_cfg.spade_version) != 2:
        raise ValueError("Stage3 requires residual SPADE version 2")
    if cfg.loss.timestep.strategy == "fixed" and int(cfg.loss.timestep.value) != int(model_cfg.timestep):
        raise ValueError("Fixed training timestep must match model.timestep used for eval")
    sd_path = str(model_cfg.sd_path)
    require_backend(str(configlib.required(cfg, "model.backend")))
    condition_type = str(configlib.required(cfg, "model.condition_type"))
    validate_sd_unet_configs(
        UNet2DConditionModel.load_config(sd_path, subfolder="unet"),
        AutoencoderKL.load_config(sd_path, subfolder="vae"),
        CLIPTextConfig.from_pretrained(sd_path, subfolder="text_encoder").to_dict(),
        DDPMScheduler.load_config(sd_path, subfolder="scheduler"),
        condition_channels=cfg.model.condition_channels if condition_type != "none" else None,
        text_dim=int(cfg.model.text_dim) if condition_type in ("deg-aware", "deg_aware_sft") else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    text_encoder.eval().requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
    vae.eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
    unet.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    lora_cfg = model_cfg.get("lora", OmegaConf.create())
    unet_rank = int(lora_cfg.get("unet_rank", 0) or 0)
    vae_encoder_rank = int(lora_cfg.get("vae_encoder_rank", 0) or 0)
    vae_decoder_rank = int(lora_cfg.get("vae_decoder_rank", 0) or 0)

    if unet_rank > 0:
        _attach_unet_lora(unet, unet_rank, str(lora_cfg.get("strategy", "full")))
        _mark_lora_trainable(unet)

    pretrained_encoder = None
    if model_cfg.get("pretrained_encoder_path"):
        pretrained_encoder = PreRestoreEncoder(
            encoder=vae.encoder,
            block_out_channels=vae.config.block_out_channels,
            cond_dim=int(configlib.required(cfg, "model.cond_dim")),
            adaln_layers=list(model_cfg.get("adaln_layers", ["down2", "down3", "mid"])),
        )
        _load_pretrained_encoder(pretrained_encoder, model_cfg.pretrained_encoder_path)
        pretrained_encoder.requires_grad_(False).eval()
        if vae_encoder_rank > 0:
            lora_config = LoraConfig(
                r=vae_encoder_rank,
                lora_alpha=vae_encoder_rank,
                init_lora_weights="gaussian",
                target_modules=r".*\.(conv1|conv2|conv_in|conv_shortcut|conv_out|to_k|to_q|to_v|to_out\.0)$",
            )
            pretrained_encoder.encoder = get_peft_model(pretrained_encoder.encoder, lora_config)
            _mark_lora_trainable(pretrained_encoder)

    if pretrained_encoder is None:
        _attach_vae_lora(vae, vae_encoder_rank, vae_decoder_rank)
    else:
        _attach_vae_lora(vae, 0, vae_decoder_rank)
    if vae_encoder_rank > 0 or vae_decoder_rank > 0:
        _mark_lora_trainable(vae)

    condition_module = _build_condition_module(cfg, unet)
    needs_deg_extractor = pretrained_encoder is not None or isinstance(
        condition_module, DegAwareConditionModule
    )
    deg_extractor = build_deg_extractor(cfg) if needs_deg_extractor else None

    if pretrained_encoder is not None:
        load_condition(deg_extractor, model_cfg.pretrained_encoder_path)

    lpips_model = None
    if float(cfg.loss.get("lambda_lpips", 0.0)) > 0:
        lpips_model = metrics.load_lpips(str(cfg.loss.get("lpips_net", "vgg")), device=None)

    model = SpadeRestorer(
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        condition_module=condition_module,
        deg_extractor=deg_extractor,
        pretrained_encoder=pretrained_encoder,
        lpips_model=lpips_model,
        timestep=int(model_cfg.timestep),
        eval_noise_seed=int(cfg.eval.noise_seed),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    all_tasks = []
    for split in ("train", "val", "test"):
        node = cfg.data.get(split)
        if node is not None:
            all_tasks.extend(OmegaConf.to_container(node, resolve=True))
    _populate_prompt_embeddings(model, all_tasks)

    if device is not None:
        model = model.to(device)
    return model


def make_optimizer(model: SpadeRestorer, cfg: OmegaConf) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Stage-3 model has no trainable parameters; enable a LoRA/condition module")
    return optim.build_optimizer(trainable, cfg.optimizer)


def set_train_mode(model: SpadeRestorer) -> None:
    utils.set_train_mode(model)


def set_eval_mode(model: SpadeRestorer) -> None:
    utils.set_eval_mode(model)


def compute_loss(
    model: SpadeRestorer,
    raw_model: SpadeRestorer,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> tuple[torch.Tensor, dict[str, float]]:
    lq = batch["lq"]
    gt = batch["gt"]
    text_embedding = raw_model.text_embedding_for(batch["task_name"])
    timestep = _sample_timestep(cfg.loss)
    prediction = model(lq, text_embedding, timestep=timestep)

    loss_l2 = F.mse_loss(prediction.float(), gt.float()) * float(cfg.loss.lambda_l2)
    loss_lpips = torch.zeros((), device=loss_l2.device)
    if float(cfg.loss.get("lambda_lpips", 0.0)) > 0:
        loss_lpips = raw_model.lpips(prediction.float(), gt.float()).mean() * float(cfg.loss.lambda_lpips)
    loss = loss_l2 + loss_lpips
    return loss, {
        "loss": float(loss.detach()),
        "loss_l2": float(loss_l2.detach()),
        "loss_lpips": float(loss_lpips.detach()),
    }


def eval_step(
    model: SpadeRestorer,
    raw_model: SpadeRestorer,
    batch: dict[str, Any],
) -> dict[str, Any]:
    text_embedding = raw_model.text_embedding_for(batch["task_name"])
    prediction = model(
        batch["lq"],
        text_embedding,
        timestep=raw_model.default_timestep,
        noise_seeds=raw_model.noise_seeds(batch["image_id"]),
    ).clamp(-1, 1)
    return {
        "pred": prediction,
        "gt": batch["gt"],
        "task_name": batch["task_name"],
    }


def validate_resume(cfg, resume_path):
    path = Path(resume_path)
    output = path.parent.parent if path.name.startswith("checkpoint-") else path
    saved = OmegaConf.load(output / "config.yaml")
    if saved.model.get("spade_version") != cfg.model.spade_version:
        raise ValueError("SPADE architecture changed; start a fresh Stage3 experiment")
    for key in ("model", "data", "loss"):
        if configlib.resume_section(saved, key) != configlib.resume_section(cfg, key):
            raise ValueError(f"Stage3 resume changes {key}; use a new experiment")
