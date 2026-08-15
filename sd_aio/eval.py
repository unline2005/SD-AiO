"""The single evaluation path shared by training and standalone eval.

Training calls :func:`run_eval` periodically; ``eval.py`` calls it for full
benchmarks.  Inference without GT shares the exact same stage forward through
:func:`run_inference`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from sd_aio import metrics, utils
from sd_aio.data import IMAGE_EXTENSIONS


@dataclass
class EvalReport:
    step: int
    task_metrics: dict[str, dict[str, float]]
    overall: dict[str, float]
    vis_paths: list[Path] = field(default_factory=list)
    classification: dict[str, Any] | None = None

    def text(self) -> str:
        lines = []
        for task_name, values in sorted(self.task_metrics.items()):
            rendered = " ".join(f"{name.upper()}={value:.4f}" for name, value in values.items())
            lines.append(f"{task_name}: {rendered}")
        if self.overall:
            rendered = " ".join(f"{name.upper()}={value:.4f}" for name, value in self.overall.items())
            lines.append(f"overall(task-equal): {rendered}")
        return f"[eval step {self.step}] " + " | ".join(lines)


def _crop_to_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, height, width = image.shape
    crop_h = (height // multiple) * multiple
    crop_w = (width // multiple) * multiple
    if crop_h < 1 or crop_w < 1:
        raise ValueError(f"Image {image.shape} is smaller than eval.crop_to_multiple={multiple}")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[:, :, top : top + crop_h, left : left + crop_w]


def _limit_batch(batch: dict[str, Any], limit: int | None) -> dict[str, Any]:
    if limit is None:
        return batch
    return {
        key: value[:limit] if isinstance(value, (torch.Tensor, list, tuple)) else value
        for key, value in batch.items()
    }


def _pad_to_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, height, width = image.shape
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


def _save_strip(
    output_dir: Path,
    step: int,
    task_name: str,
    index: int,
    lq: torch.Tensor,
    prediction: torch.Tensor,
    gt: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    strip = np.concatenate(
        [
            metrics.to_numpy_rgb(lq),
            metrics.to_numpy_rgb(prediction),
            metrics.to_numpy_rgb(gt),
        ],
        axis=1,
    )
    strip = (np.clip(strip, 0.0, 1.0) * 255.0).astype(np.uint8)
    path = output_dir / f"step_{int(step):08d}_{task_name}_{index:03d}.png"
    Image.fromarray(strip).save(path)
    return path


def _run_classifier_eval(
    stage: Any,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    loaders: OrderedDict[str, Any],
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    num_samples_per_task: int | None,
    step: int = 0,
) -> EvalReport:
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for loader in loaders.values():
            seen = 0
            for batch in loader:
                if num_samples_per_task is not None and seen >= num_samples_per_task:
                    break
                batch = _limit_batch(batch, num_samples_per_task)
                batch = utils.move_batch(batch, device, weight_dtype)
                result = stage.eval_step(model, raw_model, batch)
                all_predictions.append(result["predictions"].cpu())
                all_labels.append(result["labels"].cpu())
                seen += result["predictions"].shape[0]
    if not all_predictions:
        raise RuntimeError("Classifier eval produced no samples")
    report = stage.compute_binary_metrics(torch.cat(all_predictions), torch.cat(all_labels))
    summary = {key: value for key, value in report.items() if key != "per_class"}
    return EvalReport(
        step=step,
        task_metrics={"classification": summary},
        overall=summary,
        classification=report,
    )


def _run_image_eval(
    stage: Any,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    loaders: OrderedDict[str, Any],
    cfg: OmegaConf,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    lpips_fn: torch.nn.Module | None,
    step: int,
    output_dir: Path | None,
    save_images: bool,
    num_samples_per_task: int | None,
) -> EvalReport:
    crop_multiple = int(cfg.eval.get("crop_to_multiple", 16))
    pad_multiple = int(cfg.eval.get("pad_to_multiple", 64))
    num_images_save_eval = int(cfg.trainer.get("num_images_save_eval", 10))
    images_per_task = 0 if num_images_save_eval <= 0 else max(1, num_images_save_eval // max(1, len(loaders)))
    accumulator = metrics.MetricAccumulator()
    vis_paths: list[Path] = []

    with torch.no_grad():
        for loader in loaders.values():
            seen = 0
            for batch in loader:
                if num_samples_per_task is not None and seen >= num_samples_per_task:
                    break
                batch = _limit_batch(batch, num_samples_per_task)
                batch = utils.move_batch(batch, device, weight_dtype)
                lq = batch["lq"]
                gt = batch["gt"]
                lq_crop = _crop_to_multiple(lq, crop_multiple)
                gt_crop = _crop_to_multiple(gt, crop_multiple)
                eval_batch = {
                    **batch,
                    "lq": _pad_to_multiple(lq_crop, pad_multiple),
                    "gt": gt_crop,
                }
                result = stage.eval_step(model, raw_model, eval_batch)
                prediction = result["pred"][:, :, : lq_crop.shape[2], : lq_crop.shape[3]]
                batch_task_names = result["task_name"]

                for index, (pred, target) in enumerate(zip(prediction, gt_crop, strict=True)):
                    values = {
                        "psnr": metrics.compute_psnr(pred, target),
                        "ssim": metrics.compute_ssim(pred, target),
                    }
                    if lpips_fn is not None:
                        values["lpips"] = metrics.compute_lpips(
                            lpips_fn, pred.unsqueeze(0), target.unsqueeze(0)
                        )
                    accumulator.add(str(batch_task_names[index]), values)

                    if (
                        save_images
                        and output_dir is not None
                        and len(vis_paths) < len(loaders) * images_per_task
                    ):
                        vis_paths.append(
                            _save_strip(
                                output_dir / "eval",
                                step,
                                str(batch_task_names[index]),
                                seen + index,
                                lq_crop[index],
                                pred,
                                target,
                            )
                        )
                seen += prediction.shape[0]

    return EvalReport(
        step=step,
        task_metrics=accumulator.per_task(),
        overall=accumulator.overall(),
        vis_paths=vis_paths,
    )


def run_eval(
    stage: Any,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    loaders: OrderedDict[str, Any],
    cfg: OmegaConf,
    *,
    device: torch.device,
    weight_dtype: torch.dtype | None = None,
    lpips_fn: torch.nn.Module | None = None,
    step: int = 0,
    output_dir: str | Path | None = None,
    save_images: bool = True,
    num_samples_per_task: int | None = None,
) -> EvalReport:
    """Run one evaluation; this is the only evaluation path in the project."""
    if weight_dtype is None:
        weight_dtype = utils.weight_dtype_for(str(cfg.mixed_precision))
    stage.set_eval_mode(raw_model)

    if num_samples_per_task is None:
        num_samples_per_task = cfg.eval.get("num_samples_per_task")

    if str(cfg.stage) == "classifier":
        return _run_classifier_eval(
            stage,
            model,
            raw_model,
            loaders,
            device=device,
            weight_dtype=weight_dtype,
            num_samples_per_task=num_samples_per_task,
            step=step,
        )

    return _run_image_eval(
        stage,
        model,
        raw_model,
        loaders,
        cfg,
        device=device,
        weight_dtype=weight_dtype,
        lpips_fn=lpips_fn,
        step=step,
        output_dir=Path(output_dir) if output_dir is not None else None,
        save_images=save_images,
        num_samples_per_task=num_samples_per_task,
    )


def list_input_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No images found under {path}")
        return files
    raise FileNotFoundError(f"Input path not found: {path}")


def run_inference(
    stage: Any,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    cfg: OmegaConf,
    input_path: str | Path,
    save_dir: str | Path,
    *,
    device: torch.device,
    weight_dtype: torch.dtype | None = None,
    prompt: str | None = None,
    gt_dir: str | Path | None = None,
) -> tuple[list[Path], EvalReport | None]:
    """Pad -> shared forward -> crop back -> save.  Optionally score against a GT directory."""
    if str(cfg.stage) != "spade":
        raise ValueError(f"run_inference is only available for stage=spade, got {cfg.stage}")
    if weight_dtype is None:
        weight_dtype = utils.weight_dtype_for(str(cfg.mixed_precision))
    stage.set_eval_mode(raw_model)

    if bool(cfg.eval.get("tiling", False)):
        tile_size = int(cfg.eval.get("tile_size", 512))
        raw_model.vae.tile_sample_min_size = tile_size
        raw_model.vae.tile_latent_min_size = tile_size // 8
        raw_model.vae.enable_tiling()
    pad_multiple = int(cfg.eval.get("pad_to_multiple", 64))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if prompt is not None:
        text_embedding = raw_model.encode_prompt(prompt, device=device, dtype=torch.float32)
    else:
        if not raw_model.prompt_embeddings:
            raise RuntimeError("No cached prompt embeddings and no --prompt provided")
        text_embedding = next(iter(raw_model.prompt_embeddings.values())).to(device=device)

    accumulator = metrics.MetricAccumulator() if gt_dir is not None else None
    gt_files = (
        {p.stem: p for p in Path(gt_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        if gt_dir is not None
        else {}
    )
    input_root = Path(input_path)
    saved_paths: list[Path] = []
    with torch.no_grad():
        for image_path in tqdm(list_input_images(input_path), desc="Inference"):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            tensor = (
                torch.as_tensor(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            tensor = _pad_to_multiple(tensor, pad_multiple).to(device=device, dtype=weight_dtype)
            prediction = model(tensor, text_embedding)
            prediction = prediction[:, :, :height, :width]
            prediction_np = metrics.to_numpy_rgb(prediction[0])
            relative = image_path.relative_to(input_root) if input_root.is_dir() else Path(image_path.name)
            output_path = save_dir / relative.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.clip(prediction_np, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output_path)
            saved_paths.append(output_path)

            if gt_dir is not None and accumulator is not None:
                gt_path = gt_files.get(image_path.stem)
                if gt_path is None:
                    raise FileNotFoundError(f"No GT image matching stem {image_path.stem} in {gt_dir}")
                gt = Image.open(gt_path).convert("RGB")
                gt_tensor = torch.as_tensor(np.asarray(gt, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
                accumulator.add(
                    "inference",
                    {
                        "psnr": metrics.compute_psnr(prediction[0], gt_tensor),
                        "ssim": metrics.compute_ssim(prediction[0], gt_tensor),
                    },
                )

    report = None
    if accumulator is not None:
        report = EvalReport(
            step=0,
            task_metrics=accumulator.per_task(),
            overall=accumulator.overall(),
            vis_paths=saved_paths,
        )
    return saved_paths, report


def save_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "step": report.step,
                "task_metrics": report.task_metrics,
                "overall": report.overall,
            },
            indent=2,
        )
    )
    return path
