"""Stage 3: single-step SD 2.1 restoration with pluggable SPADE conditioning.

``SpadeRestorer`` owns the entire forward path (encode -> add noise -> denoise
-> x0 estimate -> decode), so training and inference can never drift apart.
The optional condition module lives on the UNet ResBlock ``conv2`` layers.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch import nn
from torchvision import models as torchvision_models
from transformers import AutoTokenizer, CLIPTextModel

from sd_aio import config as configlib
from sd_aio import metrics, utils
from sd_aio.classifier import build_deg_extractor
from sd_aio.vae_encoder import PreRestoreEncoder, _load_pretrained_encoder

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
        self.norm = nn.GroupNorm(num_groups=32, num_channels=output_channels)
        padding = kernel_size // 2
        self.shared = nn.Sequential(
            nn.Conv2d(cond_input_channels, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.gamma = nn.Conv2d(128, output_channels, kernel_size=kernel_size, padding=padding)
        self.beta = nn.Conv2d(128, output_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, features: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        if cond_feat.shape[2:] != features.shape[2:]:
            raise ValueError(
                f"SPADE condition {tuple(cond_feat.shape)} does not match features {tuple(features.shape)}"
            )
        normalized = self.norm(features)
        shared = self.shared(cond_feat)
        return normalized * (1.0 + self.gamma(shared)) + self.beta(shared)


class SpadeWrapper(nn.Module):
    """Replaces a ResNet ``conv2``: conv output -> SPADE modulation."""

    def __init__(self, target_module: nn.Conv2d, condition_channels: int) -> None:
        super().__init__()
        self.target_module = target_module
        self.spade = Spade(
            output_channels=self.out_channels,
            cond_input_channels=condition_channels,
        )
        self.current_cond_feat: torch.Tensor | None = None

    @property
    def _base(self) -> nn.Module:
        return getattr(self.target_module, "base_layer", self.target_module)

    @property
    def weight(self) -> nn.Parameter:
        return self._base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self._base.bias

    @property
    def kernel_size(self) -> tuple[int, ...]:
        return self._base.kernel_size

    @property
    def stride(self) -> tuple[int, ...]:
        return self._base.stride

    @property
    def padding(self) -> tuple[int, ...]:
        return self._base.padding

    @property
    def dilation(self) -> tuple[int, ...]:
        return self._base.dilation

    @property
    def groups(self) -> int:
        return self._base.groups

    @property
    def out_channels(self) -> int:
        return self._base.out_channels

    @property
    def in_channels(self) -> int:
        return self._base.in_channels

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

    @classmethod
    def inject(cls, unet: nn.Module) -> list[SpadeWrapper]:
        wrappers = []
        for _, module in unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                wrapper = cls(
                    module.conv2,
                    module.conv2.out_channels
                    if not hasattr(module.conv2, "base_layer")
                    else module.conv2.base_layer.out_channels,
                )
                module.conv2 = wrapper
                wrappers.append(wrapper)
        return wrappers


class MultiScaleExtractor(nn.Module):
    """LQ image pyramid at UNet down/mid/up scales (latent 64/32/16/8)."""

    C320, C640, C1280 = 320, 640, 1280

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
    """Spatial-only SPADE conditioning (registry key ``simple``)."""

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
        self._hooked = False

    def setup(self, unet: nn.Module) -> None:
        if self._hooked:
            return
        self._hooked = True

        def hook(resnet: ResnetBlock2D, key: str) -> None:
            conv2 = getattr(resnet, "conv2", None)
            if conv2 is None:
                return
            base = getattr(conv2, "base_layer", conv2)
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

    def get_modulation(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        f_deg: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor | None]:
        self.set_spatial_features(lq_image)
        return None, text_embedding


class DegTextFusion(nn.Module):
    """F_Deg -> one token prepended to the CLIP text embedding."""

    def __init__(self, inner_dim: int = 768, text_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(inner_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))

    def forward(self, f_deg: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        token = self.proj(f_deg).unsqueeze(1)
        return torch.cat([token, text_embedding], dim=1)


class DegAwareConditionModule(SpadeConditionModule):
    """SPADE spatial modulation + degradation token for cross-attention (``deg-aware``)."""

    def __init__(
        self,
        backbone_type: BackboneType = "simple-conv",
        inner_dim: int = 768,
        text_dim: int = 1024,
        channel_dims: tuple[int, int, int] = (320, 640, 1280),
    ) -> None:
        super().__init__(backbone_type, channel_dims=channel_dims)
        self.text_fusion = DegTextFusion(inner_dim=inner_dim, text_dim=text_dim)

    def get_modulation(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        f_deg: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor | None]:
        self.set_spatial_features(lq_image)
        if text_embedding is not None and f_deg is not None:
            text_embedding = self.text_fusion(f_deg, text_embedding)
        return None, text_embedding


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


def _attach_vae_lora(vae: nn.Module, rank: int, part: Literal["encoder", "decoder"]) -> None:
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=rf"^{part}{VAE_LORA_TARGET}",
    )
    vae.add_adapter(config, adapter_name=f"restoration_{part}")


def _mark_lora_trainable(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = "lora" in name


def _sample_timestep(loss_cfg: OmegaConf) -> int:
    timestep_cfg = loss_cfg.get("timestep", OmegaConf.create({"strategy": "fixed", "value": 100}))
    if timestep_cfg.get("strategy", "fixed") == "range":
        low, high = timestep_cfg["range"]
        return int(torch.randint(int(low), int(high) + 1, (1,)).item())
    return int(timestep_cfg.get("value", 100))


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
        self.scaling_factor = float(vae.config.scaling_factor)
        self.prompt_embeddings = nn.ParameterDict()
        # Kept as plain attributes (not registered submodules) so the text
        # encoder stays on CPU and never gets pulled onto GPU by model.to().
        self._aux: dict[str, Any] = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }

    # ------------------------------------------------------------------ helpers
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

    # ------------------------------------------------------------------ encode/decode
    def encode_lq(self, lq_image: torch.Tensor, f_deg: torch.Tensor | None = None) -> torch.Tensor:
        if self.pretrained_encoder is not None:
            if f_deg is None:
                f_deg = self.deg_extractor(lq_image) if self.deg_extractor is not None else None
            if f_deg is None:
                raise RuntimeError("pretrained_encoder requires a degradation feature extractor")
            z_raw = self.pretrained_encoder(lq_image, f_deg)
            z_mean = self.vae.quant_conv(z_raw)[:, :4]
            return z_mean * self.scaling_factor
        posterior = self.vae.encode(lq_image).latent_dist
        return posterior.sample() * self.scaling_factor

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.scaling_factor).sample.clamp(-1.0, 1.0)

    @staticmethod
    def eps_to_coeff(timesteps: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
        alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
        alphas = alphas.view(-1, 1, 1, 1)
        return ((1.0 - alphas) ** 0.5) / (alphas**0.5)

    # ------------------------------------------------------------------ one forward for train + eval + inference
    def forward(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        timestep: int | None = None,
    ) -> torch.Tensor:
        f_deg = self.deg_extractor(lq_image) if self.deg_extractor is not None else None
        z0 = self.encode_lq(lq_image, f_deg=f_deg)
        noise = torch.randn_like(z0)
        if timestep is None:
            timestep = self.default_timestep
        timesteps = torch.full((z0.shape[0],), int(timestep), device=z0.device, dtype=torch.long)
        z_t = self.noise_scheduler.add_noise(z0, noise, timesteps)

        if self.condition_module is not None:
            _, text_embedding = self.condition_module.get_modulation(lq_image, text_embedding, f_deg=f_deg)

        if text_embedding is None:
            raise RuntimeError("SpadeRestorer.forward requires text_embedding")
        noise_pred = self.unet(z_t, timesteps, encoder_hidden_states=text_embedding).sample
        coeff = self.eps_to_coeff(timesteps, self.noise_scheduler).to(device=z0.device, dtype=z0.dtype)
        denoised = z0 + coeff * (noise - noise_pred)
        return self.decode_latent(denoised)

    def sample(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor,
        timestep: int | None = None,
    ) -> torch.Tensor:
        return self.forward(lq_image, text_embedding, timestep=timestep)


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
    inner_dim = int(configlib.required(cfg, "model.cond_dim"))
    text_dim = int(cfg.model.get("text_dim", 1024))
    channel_dims = tuple(int(dim) for dim in cfg.model.get("condition_channels", [320, 640, 1280]))
    if condition_type == "simple":
        module = SpadeConditionModule(backbone_type=backbone, channel_dims=channel_dims)
    elif condition_type in ("deg-aware", "deg_aware_sft"):
        module = DegAwareConditionModule(
            backbone_type=backbone,
            inner_dim=inner_dim,
            text_dim=text_dim,
            channel_dims=channel_dims,
        )
    else:
        raise ValueError(f"Unknown condition_type {condition_type}; use none/simple/deg-aware")
    module.setup(unet)
    return module


def build_model(cfg: OmegaConf, device: torch.device | None = None) -> SpadeRestorer:
    model_cfg = cfg.model
    sd_path = str(model_cfg.sd_path)

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

    # LoRA first: later SPADE wraps the (possibly LoRA-wrapped) conv2 layer.
    if unet_rank > 0:
        _attach_unet_lora(unet, unet_rank, str(lora_cfg.get("strategy", "full")))
        _mark_lora_trainable(unet)

    pretrained_encoder = None
    if model_cfg.get("pretrained_encoder_path"):
        pretrained_encoder = PreRestoreEncoder(
            encoder=vae.encoder,
            block_out_channels=vae.config.block_out_channels,
            cond_dim=int(model_cfg.get("cond_dim", 768)),
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

    if pretrained_encoder is None and vae_encoder_rank > 0:
        _attach_vae_lora(vae, vae_encoder_rank, "encoder")
        _mark_lora_trainable(vae)
    if vae_decoder_rank > 0:
        _attach_vae_lora(vae, vae_decoder_rank, "decoder")
        _mark_lora_trainable(vae)

    condition_module = _build_condition_module(cfg, unet)
    needs_deg_extractor = condition_module is not None and isinstance(
        condition_module, DegAwareConditionModule
    )
    needs_deg_extractor = needs_deg_extractor or pretrained_encoder is not None
    deg_extractor = build_deg_extractor(cfg, device=None) if needs_deg_extractor else None

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
        timestep=int(model_cfg.get("timestep", 100)),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    all_tasks = []
    for split in ("train", "test"):
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
    optimizer_cfg = cfg.optimizer
    return torch.optim.AdamW(
        trainable,
        lr=float(optimizer_cfg.lr),
        betas=(float(optimizer_cfg.betas[0]), float(optimizer_cfg.betas[1])),
        weight_decay=float(optimizer_cfg.weight_decay),
        eps=float(optimizer_cfg.eps),
    )


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
    cfg: OmegaConf,
) -> dict[str, Any]:
    text_embedding = raw_model.text_embedding_for(batch["task_name"])
    prediction = model(batch["lq"], text_embedding, timestep=raw_model.default_timestep)
    return {
        "pred": prediction,
        "gt": batch["gt"],
        "lq": batch["lq"],
        "task_name": batch["task_name"],
    }
