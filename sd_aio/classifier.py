"""Stage 1: DINOv2 degradation classifier.

Self-contained module following the shared stage protocol::

    build_model(cfg, device) -> model
    make_optimizer(model, cfg) -> optimizer
    set_train_mode(model)
    set_eval_mode(model)
    compute_loss(model, raw_model, batch, cfg) -> (loss, logs)
    eval_step(model, raw_model, batch, cfg) -> sample dict

``model`` is the accelerator-prepared model (DDP/autocast aware) and
``raw_model`` is its unwrapped counterpart used for attribute access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from sd_aio import checkpoint, utils


class ClassifierHead(nn.Module):
    """LayerNorm + MLP producing ``[B, num_classes, 2]`` binary logits."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim or feature_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim or feature_dim // 2, num_classes * 2),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(features)).view(features.shape[0], self.num_classes, 2)


class DegradationClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dino_path: str | None = None,
        freeze_encoder: bool = True,
        head_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dino_path is None:
            self.encoder = Dinov2Model(Dinov2Config())
        elif Path(dino_path).is_dir():
            self.encoder = Dinov2Model.from_pretrained(dino_path)
        else:
            self.encoder = Dinov2Model.from_pretrained(dino_path)

        self.feature_dim = int(self.encoder.config.hidden_size)
        self.head = ClassifierHead(self.feature_dim, num_classes, head_hidden_dim)
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)

    def forward_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token, self.head(cls_token)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_features(pixel_values)[1]

    def class_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)[:, :, 0]


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
        self.inner_dim = int(inner_dim or classifier.feature_dim)
        self.deg_embedding = nn.Parameter(torch.empty(num_classes, self.inner_dim))
        nn.init.orthogonal_(self.deg_embedding)
        self.deg_alpha = nn.Parameter(torch.tensor(10.0))

    def set_trainable_embedding(self, trainable: bool) -> None:
        self.deg_embedding.requires_grad_(trainable)
        self.deg_alpha.requires_grad_(trainable)

    def forward_features(self, lq_images: torch.Tensor) -> torch.Tensor:
        classifier = self.classifier
        classifier_dtype = next(classifier.parameters()).dtype
        with torch.no_grad():
            cls_token, logits = classifier.forward_features(lq_images.to(dtype=classifier_dtype))
            probabilities = classifier.class_probabilities(logits).to(dtype=lq_images.dtype)
            cls_token = cls_token.to(dtype=lq_images.dtype)
        # Embedding projection stays differentiable when train_deg_embedding is enabled.
        # Cast to the caller's dtype so the projection is safe both inside and
        # outside an autocast region (Stage 2 calls this module directly).
        embedding = self.deg_embedding.to(dtype=lq_images.dtype)
        alpha = self.deg_alpha.to(dtype=lq_images.dtype)
        return cls_token + alpha * (probabilities @ embedding)

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(lq_images)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-label focal loss on the positive-class logit of each ``[B, C, 2]`` head."""
    logits = logits[..., 0]
    bce = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype), reduction="none")
    probability = torch.exp(-bce)
    loss = (1.0 - probability) ** gamma * bce
    if alpha is not None:
        loss = loss * alpha.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


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
    model = DegradationClassifier(
        num_classes=int(model_cfg.num_deg_types),
        dino_path=_required_dino_path(cfg),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
    )
    if device is not None:
        model = model.to(device)
    return model


def make_optimizer(model: DegradationClassifier, cfg: OmegaConf) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.optimizer
    backbone_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    head_parameters = [p for p in model.head.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(optimizer_cfg.head_lr)})
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": float(optimizer_cfg.backbone_lr)})
    if not groups:
        raise RuntimeError("Classifier has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        betas=(float(optimizer_cfg.betas[0]), float(optimizer_cfg.betas[1])),
        weight_decay=float(optimizer_cfg.weight_decay),
        eps=float(optimizer_cfg.eps),
    )


def set_train_mode(model: DegradationClassifier) -> None:
    utils.set_train_mode(model)


def set_eval_mode(model: DegradationClassifier) -> None:
    utils.set_eval_mode(model)


def compute_loss(
    model: DegradationClassifier,
    raw_model: DegradationClassifier,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(batch["lq"])
    loss = focal_loss(logits, batch["label"], gamma=float(cfg.loss.focal_gamma))
    with torch.no_grad():
        predictions = (logits[..., 0] > 0.0).long()
        accuracy = float((predictions == batch["label"].long()).float().mean())
    return loss, {"loss": float(loss.detach()), "accuracy": accuracy}


def eval_step(
    model: DegradationClassifier,
    raw_model: DegradationClassifier,
    batch: dict[str, Any],
    cfg: OmegaConf,
) -> dict[str, Any]:
    logits = model(batch["lq"])
    return {
        "predictions": (logits[..., 0] > 0.0).long(),
        "labels": batch["label"].long(),
        "task_name": batch["task_name"],
    }


def build_deg_extractor(cfg: OmegaConf, device: torch.device | None = None) -> DegFeatureExtractor:
    """Build the frozen F_Deg extractor shared by Stage 2 and Stage 3."""
    model_cfg = cfg.model
    classifier = DegradationClassifier(
        num_classes=int(model_cfg.num_deg_types),
        dino_path=_required_dino_path(cfg),
        freeze_encoder=True,
    )
    checkpoint_path = model_cfg.get("degradation_classifier_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            checkpoint.load_model_weights(classifier, checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            missing, unexpected = classifier.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"Classifier checkpoint has unexpected keys: {unexpected[:5]}")
            if missing:
                print(
                    f"  [deg-extractor] classifier checkpoint missing {len(missing)} keys (frozen parts are fine)"
                )

    inner_dim = model_cfg.get("cond_dim") or classifier.feature_dim
    extractor = DegFeatureExtractor(classifier, int(model_cfg.num_deg_types), int(inner_dim))
    extractor.classifier.requires_grad_(False).eval()
    extractor.set_trainable_embedding(bool(model_cfg.get("train_deg_embedding", False)))
    if device is not None:
        extractor = extractor.to(device)
    return extractor
