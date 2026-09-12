"""DINOv2 multi-label degradation classifier and its feature interface."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from sd_aio import checkpoint, optim, utils
from sd_aio.config import DEFAULTS_PATH, required

_DEFAULTS = OmegaConf.load(DEFAULTS_PATH)


class ClassifierHead(nn.Module):
    """LayerNorm + MLP producing one independent logit per degradation."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(feature_dim)
        hidden_dim = feature_dim // 2 if hidden_dim is None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(features))


class LabelQueryHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("query hidden_dim must be positive and divisible by num_heads")
        self.norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward_with_evidence(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.projection(self.norm(tokens))
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        evidence, _ = self.attention(queries, tokens, tokens, need_weights=False)
        evidence = self.output_norm(evidence)
        return evidence, (evidence * self.weight).sum(-1) + self.bias

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_evidence(tokens)[1]


class PatchPredictionHead(nn.Module):
    """Fuse DINO block outputs and predict a spatial restoration target."""

    TARGETS: ClassVar[set[str]] = {"mean_abs", "mask_fraction", "residual"}

    def __init__(
        self,
        feature_dim: int,
        layers: list[int],
        hidden_dim: int,
        patch_size: int,
        target: str,
    ) -> None:
        super().__init__()
        if target not in self.TARGETS:
            raise ValueError(f"Unknown patch target {target!r}; use {sorted(self.TARGETS)}")
        if not layers or len(layers) != len(set(layers)) or any(layer < 1 for layer in layers):
            raise ValueError("Patch layers must be unique positive one-based block indices")
        if hidden_dim < 1:
            raise ValueError("Patch hidden_dim must be positive")
        self.layers = tuple(layers)
        self.patch_size = patch_size
        self.target = target
        self.norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in layers)
        self.fuse = nn.Conv2d(feature_dim * len(layers), hidden_dim, 1)
        self.mix = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        output_channels = 3 * patch_size**2 if target == "residual" else 1
        self.output = nn.Conv2d(hidden_dim, output_channels, 1)

    def forward(self, hidden_states: tuple[torch.Tensor, ...], grid: tuple[int, int]) -> torch.Tensor:
        height, width = grid
        features = []
        for layer, norm in zip(self.layers, self.norms, strict=True):
            tokens = norm(hidden_states[layer][:, 1:])
            if tokens.shape[1] != height * width:
                raise ValueError(
                    f"DINO block {layer} has {tokens.shape[1]} patches, expected {height * width}"
                )
            features.append(tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width))
        fused = F.gelu(self.fuse(torch.cat(features, dim=1)))
        prediction = self.output(F.gelu(self.mix(fused)))
        return F.pixel_shuffle(prediction, self.patch_size) if self.target == "residual" else prediction


class DegradationClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dino_path: str | None = None,
        freeze_encoder: bool = True,
        head_hidden_dim: int | None = None,
        head_type: str = "mlp",
        query_heads: int = 4,
        train_last_blocks: int | None = None,
        l2sp_weight: float = 0.0,
        patch_target: str = "none",
        patch_layers: list[int] | None = None,
        patch_hidden_dim: int = 128,
        pad_to_patch_multiple: bool = False,
    ) -> None:
        super().__init__()
        if dino_path is None:
            self.encoder = Dinov2Model(Dinov2Config())
        else:
            self.encoder = Dinov2Model.from_pretrained(dino_path)

        # Classification never masks input patches, so this parameter is unused.
        if self.encoder.embeddings.mask_token is not None:
            self.encoder.embeddings.mask_token.requires_grad_(False)
        self.feature_dim = int(self.encoder.config.hidden_size)
        self.patch_size = int(self.encoder.config.patch_size)
        self.pad_to_patch_multiple = pad_to_patch_multiple
        self.head_type = head_type
        if head_type == "mlp":
            self.head = ClassifierHead(self.feature_dim, num_classes, head_hidden_dim)
        elif head_type == "query":
            self.head = LabelQueryHead(self.feature_dim, num_classes, head_hidden_dim, query_heads)
        else:
            raise ValueError(f"Unknown classifier head: {head_type}")
        self.patch_head = None
        if patch_target != "none":
            layers = list(patch_layers or [])
            depth = int(self.encoder.config.num_hidden_layers)
            if any(layer > depth for layer in layers):
                raise ValueError(f"Patch layers {layers} exceed DINO depth {depth}")
            self.patch_head = PatchPredictionHead(
                self.feature_dim, layers, patch_hidden_dim, self.patch_size, patch_target
            )
        if train_last_blocks is not None:
            layers = self.encoder.encoder.layer
            if freeze_encoder or not 1 <= train_last_blocks <= len(layers):
                raise ValueError("train_last_blocks requires an unfrozen encoder and a valid block count")
            self.encoder.requires_grad_(False)
            for layer in layers[-train_last_blocks:]:
                layer.requires_grad_(True)
            self.encoder.layernorm.requires_grad_(True)
        self.l2sp_weight = l2sp_weight
        if l2sp_weight < 0:
            raise ValueError("l2sp_weight must be nonnegative")
        self._reference_names = []
        if l2sp_weight:
            for name, parameter in self.encoder.named_parameters():
                if parameter.requires_grad and not freeze_encoder:
                    self.register_buffer(
                        f"_l2sp_{len(self._reference_names)}", parameter.detach().clone(), persistent=False
                    )
                    self._reference_names.append(name)
        if freeze_encoder:
            self.encoder.requires_grad_(False).eval()

    def _encoder_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.pad_to_patch_multiple:
            pad_h = (-pixel_values.shape[-2]) % self.patch_size
            pad_w = (-pixel_values.shape[-1]) % self.patch_size
            if pad_h or pad_w:
                pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="replicate")
        # All stages provide RGB in [-1, 1]; DINOv2 expects ImageNet normalization.
        image = (pixel_values + 1.0) * 0.5
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (image - mean) / std

    def forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=self._encoder_input(pixel_values)).last_hidden_state

    def forward_with_patch(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_head is None:
            raise RuntimeError("Patch prediction requested without a patch head")
        encoder_input = self._encoder_input(pixel_values)
        outputs = self.encoder(pixel_values=encoder_input, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.head(outputs.last_hidden_state[:, 1:] if self.head_type == "query" else cls_token)
        grid = (encoder_input.shape[-2] // self.patch_size, encoder_input.shape[-1] // self.patch_size)
        return logits, self.patch_head(outputs.hidden_states, grid)

    def forward_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.forward_tokens(pixel_values)
        cls_token = tokens[:, 0]
        return cls_token, self.head(tokens[:, 1:] if self.head_type == "query" else cls_token)

    def forward_evidence(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_type != "query":
            raise ValueError("Label evidence requires a query classifier head")
        return self.head.forward_with_evidence(self.forward_tokens(pixel_values)[:, 1:])

    def forward(
        self, pixel_values: torch.Tensor, *, with_patch: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if with_patch:
            return self.forward_with_patch(pixel_values)
        return self.forward_features(pixel_values)[1]


class DegFeatureExtractor(nn.Module):
    """F_Deg = cls_token + alpha * (per-class probabilities @ deg_embedding).

    Used as the frozen conditioning feature by Stage 2 and Stage 3.  The
    classifier itself is frozen; ``deg_embedding`` and ``deg_alpha`` are
    optionally trainable (configurable per stage).
    """

    def __init__(
        self,
        classifier: DegradationClassifier,
        num_classes: int,
        inner_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.deg_embedding = nn.Parameter(torch.empty(num_classes, int(inner_dim or classifier.feature_dim)))
        nn.init.orthogonal_(self.deg_embedding)
        self.deg_alpha = nn.Parameter(torch.tensor(10.0))

    def set_trainable_embedding(self, trainable: bool) -> None:
        self.deg_embedding.requires_grad_(trainable)
        self.deg_alpha.requires_grad_(trainable)

    def forward_features(self, lq_images: torch.Tensor) -> torch.Tensor:
        classifier = self.classifier
        classifier_parameter = next(classifier.parameters())
        with torch.no_grad():
            cls_token, logits = classifier.forward_features(
                lq_images.to(device=classifier_parameter.device, dtype=classifier_parameter.dtype)
            )
            cls_token = cls_token.to(device=lq_images.device, dtype=lq_images.dtype)
            probabilities = torch.sigmoid(logits).to(device=lq_images.device, dtype=lq_images.dtype)
        embedding = self.deg_embedding.to(device=lq_images.device, dtype=lq_images.dtype)
        alpha = self.deg_alpha.to(device=lq_images.device, dtype=lq_images.dtype)
        return cls_token + alpha * (probabilities @ embedding)

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(lq_images)


class QueryEvidenceCondition(nn.Module):
    def __init__(self, classifier: DegradationClassifier, cond_dim: int, weights_path: Path):
        super().__init__()
        if classifier.head_type != "query":
            raise ValueError("query_evidence requires a query classifier")
        expected = classifier.head.queries.numel()
        if cond_dim != expected:
            raise ValueError(f"cond_dim={cond_dim}, expected K * evidence_dim = {expected}")
        self.classifier = classifier.requires_grad_(False).eval()
        with weights_path.open("rb") as stream:
            digest = hashlib.sha256(stream.read()).digest()
        self.register_buffer(
            "classifier_sha256", torch.tensor(list(digest), dtype=torch.uint8), persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        parameter = next(self.classifier.parameters())
        with torch.no_grad():
            evidence, logits = self.classifier.forward_evidence(images.to(dtype=parameter.dtype))
            condition = (logits.sigmoid().unsqueeze(-1) * evidence).flatten(1)
        return condition.to(dtype=images.dtype)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != labels.shape:
        raise ValueError(f"Expected matching [B, K] logits/labels, got {logits.shape} and {labels.shape}")
    bce = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype), reduction="none")
    return ((1.0 - torch.exp(-bce)) ** gamma * bce).mean()


def asymmetric_loss(logits, labels, gamma_pos, gamma_neg, clip):
    if logits.ndim != 2 or logits.shape != labels.shape:
        raise ValueError("Expected matching [B, K] logits and labels")
    if gamma_pos < 0 or gamma_neg < 0 or not 0 <= clip < 1:
        raise ValueError("ASL requires nonnegative gammas and clip in [0, 1)")
    logits, labels = logits.float(), labels.float()
    positive = logits.sigmoid()
    negative = (1 - positive + clip).clamp(max=1)
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits) if clip == 0 else negative.clamp_min(1e-8).log()
    with torch.no_grad():
        probability = positive * labels + negative * (1 - labels)
        gamma = gamma_pos * labels + gamma_neg * (1 - labels)
        weight = (1 - probability).pow(gamma)
    return -(weight * (labels * log_positive + (1 - labels) * log_negative)).mean()


def l2sp_loss(model):
    parameters = dict(model.encoder.named_parameters())
    loss = next(model.head.parameters()).new_zeros((), dtype=torch.float32)
    for index, name in enumerate(model._reference_names):
        delta = parameters[name].float() - getattr(model, f"_l2sp_{index}").float()
        loss = loss + delta.square().sum()
    return loss


def before_optimizer_step(model, cfg, step):
    warmup = int(_classifier_options(cfg).head_warmup_steps)
    if warmup < 0:
        raise ValueError("head_warmup_steps must be nonnegative")
    if step < warmup:
        for parameter in model.encoder.parameters():
            parameter.grad = None


def _classifier_options(cfg):
    defaults = _DEFAULTS.model.classifier
    return OmegaConf.merge(defaults, OmegaConf.select(cfg, "model.classifier") or {})


def compute_binary_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predictions = predictions.long()
    labels = labels.long()
    num_classes = labels.shape[1]
    per_class = []
    for class_index in range(num_classes):
        prediction = predictions[:, class_index]
        label = labels[:, class_index]
        tp = int(((prediction == 1) & (label == 1)).sum())
        fp = int(((prediction == 1) & (label == 0)).sum())
        fn = int(((prediction == 0) & (label == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class.append(
            {
                "accuracy": float((prediction == label).float().mean()),
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            }
        )
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "exact_match": float((predictions == labels).all(dim=1).float().mean()),
        "precision": float(sum(item["precision"] for item in per_class) / max(num_classes, 1)),
        "recall": float(sum(item["recall"] for item in per_class) / max(num_classes, 1)),
        "f1": float(sum(item["f1"] for item in per_class) / max(num_classes, 1)),
        "per_class": per_class,
    }


def _required_dino_path(cfg: OmegaConf) -> str:
    dino_path = cfg.model.get("dino_path")
    if dino_path is None:
        raise ValueError("model.dino_path is required for classifier / F_Deg extractor stages")
    return str(dino_path)


def build_model(cfg: OmegaConf, device: torch.device | None = None) -> DegradationClassifier:
    model_cfg = cfg.model
    options = _classifier_options(cfg)
    patch = options.patch
    model = DegradationClassifier(
        num_classes=int(model_cfg.num_deg_types),
        dino_path=_required_dino_path(cfg),
        freeze_encoder=bool(model_cfg.freeze_encoder),
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        head_type=str(options.head_type),
        query_heads=int(options.query_heads),
        train_last_blocks=options.train_last_blocks,
        l2sp_weight=float(options.l2sp_weight),
        patch_target=str(patch.target),
        patch_layers=list(patch.layers),
        patch_hidden_dim=int(patch.hidden_dim),
        pad_to_patch_multiple=bool(options.pad_to_patch_multiple),
    )
    if device is not None:
        model = model.to(device)
    return model


def make_optimizer(model: DegradationClassifier, cfg: OmegaConf) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.optimizer
    backbone_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    head_parameters = [p for p in model.head.parameters() if p.requires_grad]
    if model.patch_head is not None:
        head_parameters.extend(p for p in model.patch_head.parameters() if p.requires_grad)
    groups: list[dict[str, Any]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(optimizer_cfg.head_lr)})
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": float(optimizer_cfg.backbone_lr)})
    if not groups:
        raise RuntimeError("Classifier has no trainable parameters")
    return optim.build_optimizer(groups, optimizer_cfg)


def set_train_mode(model: DegradationClassifier) -> None:
    # Set each component explicitly so eval -> train also restores DINO dropout.
    model.training = True
    model.head.train()
    if model.patch_head is not None:
        model.patch_head.train()
    model.encoder.train(any(parameter.requires_grad for parameter in model.encoder.parameters()))
    if any(
        not p.requires_grad
        for p in model.encoder.embeddings.parameters()
        if p is not model.encoder.embeddings.mask_token
    ):
        model.encoder.embeddings.eval()
    for layer in model.encoder.encoder.layer:
        if not any(p.requires_grad for p in layer.parameters()):
            layer.eval()


def set_eval_mode(model: DegradationClassifier) -> None:
    utils.set_eval_mode(model)


def compute_loss(
    model: DegradationClassifier,
    raw_model: DegradationClassifier,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> tuple[torch.Tensor, dict[str, float]]:
    patch_cfg = _classifier_options(cfg).patch
    patch_target = str(patch_cfg.target)
    if patch_target == "none":
        logits = model(batch["lq"])
        patch_prediction = None
    else:
        logits, patch_prediction = model(batch["lq"], with_patch=True)
    loss_cfg = OmegaConf.merge(_DEFAULTS.loss, cfg.loss)
    if loss_cfg.classification == "bce":
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"].to(dtype=logits.dtype))
    elif loss_cfg.classification == "asl":
        loss = asymmetric_loss(
            logits, batch["label"], loss_cfg.asl.gamma_pos, loss_cfg.asl.gamma_neg, loss_cfg.asl.clip
        )
    elif loss_cfg.classification == "focal":
        loss = focal_loss(logits, batch["label"], gamma=float(required(cfg, "loss.focal_gamma")))
    elif loss_cfg.classification == "bce":
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"].to(dtype=logits.dtype))
    else:
        raise ValueError(f"Unknown classification loss: {loss_cfg.classification}")
    classification_loss = loss.detach()
    patch_loss = loss.new_zeros(())
    if patch_prediction is not None:
        patch_loss = _patch_loss(
            patch_prediction,
            batch["lq"],
            batch["gt"],
            patch_target,
            raw_model.patch_size,
            float(loss_cfg.patch.mask_threshold),
        )
        loss = loss + float(loss_cfg.patch.weight) * patch_loss
    regularization = l2sp_loss(raw_model)
    loss = loss + raw_model.l2sp_weight * regularization
    with torch.no_grad():
        predictions = (torch.sigmoid(logits) > 0.5).long()
        accuracy = float((predictions == batch["label"].long()).float().mean())
    return loss, {
        "loss": float(loss.detach()),
        "classification_loss": float(classification_loss),
        "patch_loss": float(patch_loss.detach()),
        "l2sp": float(regularization.detach()),
        "accuracy": accuracy,
    }


def _patch_loss(
    prediction: torch.Tensor,
    lq: torch.Tensor,
    gt: torch.Tensor,
    target: str,
    patch_size: int,
    mask_threshold: float,
) -> torch.Tensor:
    if lq.shape != gt.shape:
        raise ValueError(f"Patch supervision requires matching LQ/GT tensors, got {lq.shape} and {gt.shape}")
    height, width = lq.shape[-2:]
    pad_h, pad_w = (-height) % patch_size, (-width) % patch_size
    difference = (lq.float() - gt.float()).abs().mean(dim=1, keepdim=True) * 0.5
    if target == "residual":
        residual = (lq.float() - gt.float()) * 0.5
        return F.l1_loss(prediction[..., :height, :width].float(), residual)
    if target not in {"mean_abs", "mask_fraction"}:
        raise ValueError(f"Unknown patch target: {target}")
    values = difference if target == "mean_abs" else (difference > mask_threshold).float()
    valid = torch.ones_like(values)
    if pad_h or pad_w:
        values = F.pad(values, (0, pad_w, 0, pad_h))
        valid = F.pad(valid, (0, pad_w, 0, pad_h))
    pooled = F.avg_pool2d(values, patch_size, patch_size)
    fraction = F.avg_pool2d(valid, patch_size, patch_size)
    patch_target = pooled / fraction.clamp_min(1e-8)
    if prediction.shape != patch_target.shape:
        raise ValueError(f"Patch prediction {prediction.shape} does not match target {patch_target.shape}")
    if target == "mean_abs":
        return F.l1_loss(prediction.float(), patch_target)
    return F.binary_cross_entropy_with_logits(prediction.float(), patch_target)


def eval_step(
    model: DegradationClassifier,
    raw_model: DegradationClassifier,
    batch: dict[str, Any],
) -> dict[str, Any]:
    logits = model(batch["lq"])
    return {
        "predictions": (torch.sigmoid(logits) > 0.5).long(),
        "labels": batch["label"].long(),
    }


def build_deg_extractor(cfg: OmegaConf) -> DegFeatureExtractor | QueryEvidenceCondition:
    """Build the frozen F_Deg extractor shared by Stage 2 and Stage 3."""
    model_cfg = cfg.model
    mode = model_cfg.get("condition_mode", _DEFAULTS.model.condition_mode)
    if mode not in {"legacy", "query_evidence"}:
        raise ValueError(f"Unknown condition mode: {mode}")
    options = _classifier_options(cfg)
    if mode == "query_evidence" and (options.head_type != "query" or model_cfg.train_deg_embedding):
        raise ValueError("query_evidence requires a frozen query classifier, without random embedding")
    classifier = DegradationClassifier(
        num_classes=int(model_cfg.num_deg_types),
        dino_path=_required_dino_path(cfg),
        freeze_encoder=mode == "legacy",
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        head_type=str(options.head_type),
        query_heads=int(options.query_heads),
        train_last_blocks=options.train_last_blocks if mode == "query_evidence" else None,
    )
    checkpoint_path = model_cfg.get("degradation_classifier_path")
    if checkpoint_path is None:
        raise ValueError("model.degradation_classifier_path is required for F_Deg extraction")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        checkpoint.load_model_weights(classifier, checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = classifier.load_state_dict(state, strict=False)
        trainable = {name for name, parameter in classifier.named_parameters() if parameter.requires_grad}
        if unexpected or (trainable & set(missing)):
            raise RuntimeError(
                f"Classifier checkpoint mismatch: unexpected={unexpected[:5]} "
                f"missing_trainable={sorted(trainable & set(missing))[:5]}"
            )

    if mode == "query_evidence":
        return QueryEvidenceCondition(classifier, int(model_cfg.cond_dim), checkpoint_path)
    extractor = DegFeatureExtractor(classifier, int(model_cfg.num_deg_types), int(model_cfg.cond_dim))
    extractor.classifier.requires_grad_(False).eval()
    extractor.set_trainable_embedding(bool(model_cfg.get("train_deg_embedding", False)))
    return extractor


def save_tsne_visualization(
    model: DegradationClassifier,
    loaders: dict[str, Any],
    cfg: OmegaConf,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    output_dir: str | Path,
    step: int,
) -> Path:
    """Extract fixed validation CLS features and render each data source separately."""
    import os
    import subprocess

    import numpy as np

    tsne_cfg = cfg.eval.tsne
    samples_per_task = int(tsne_cfg.samples_per_task)
    if samples_per_task < 1:
        raise ValueError("eval.tsne.samples_per_task must be positive")
    features = []
    labels = []
    sources = []
    model.eval()
    with torch.no_grad():
        for task_name, loader in loaders.items():
            seen = 0
            for batch in loader:
                if seen >= samples_per_task:
                    break
                limit = min(batch["lq"].shape[0], samples_per_task - seen)
                images = batch["lq"][:limit].to(device=device, dtype=weight_dtype)
                cls_token, _ = model.forward_features(images)
                features.append(cls_token.float().cpu().numpy())
                labels.append(batch["label"][:limit].long().numpy())
                source = next((name for name in ("FoundIR", "GGT", "CDD11") if name in task_name), task_name)
                sources.extend([source] * limit)
                seen += limit
    if not features:
        raise RuntimeError("t-SNE feature extraction produced no samples")
    target = Path(output_dir) / "tsne" / f"step_{step:08d}"
    target.mkdir(parents=True, exist_ok=True)
    feature_path = target / "features.npz"
    np.savez_compressed(
        feature_path,
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        sources=np.asarray(sources),
        class_names=np.asarray(list(cfg.data.deg_types)),
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "4"
    subprocess.run(
        [
            str(tsne_cfg.plot_python),
            str(tsne_cfg.plot_script),
            str(feature_path),
            str(target),
            "--perplexity",
            str(tsne_cfg.perplexity),
            "--seed",
            str(tsne_cfg.seed),
            "--iterations",
            str(tsne_cfg.iterations),
        ],
        check=True,
        env=environment,
    )
    return target
