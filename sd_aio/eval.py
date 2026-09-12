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
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from sd_aio import metrics, utils
from sd_aio.data import IMAGE_EXTENSIONS, PairedTransform, preprocessing_options


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
    extra_panels: dict[str, torch.Tensor] | None = None,
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
    if extra_panels is None:
        Image.fromarray(strip).save(path)
    else:
        panels = [
            ("LQ", lq),
            ("Original VAE (LQ)", extra_panels["baseline"]),
            ("Pre-restored", prediction),
            ("Original VAE (GT)", extra_panels["vae_gt"]),
            ("GT", gt),
        ]
        height, width = gt.shape[-2:]
        header = max(28, width // 24)
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, width // 32))
        canvas = Image.new("RGB", (width * len(panels), height + header), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (label, tensor) in enumerate(panels):
            array = (np.clip(metrics.to_numpy_rgb(tensor), 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (column * width, header))
            draw.text((column * width + 8, 7), label, fill="black", font=font)
        canvas.save(path)
    return path


def _run_classifier_eval(
    stage: Any,
    raw_model: torch.nn.Module,
    loaders: OrderedDict[str, Any],
    cfg: OmegaConf,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    num_samples_per_task: int | None,
    step: int = 0,
) -> EvalReport:
    class_names = [str(name) for name in cfg.data.deg_types]
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for loader in loaders.values():
            seen = 0
            for batch in loader:
                remaining = None if num_samples_per_task is None else num_samples_per_task - seen
                if remaining is not None and remaining <= 0:
                    break
                batch = _limit_batch(batch, remaining)
                batch = utils.move_batch(batch, device, weight_dtype)
                # Only rank 0 evaluates: bypass DDP forward and its buffer broadcasts.
                result = stage.eval_step(raw_model, raw_model, batch)
                all_predictions.append(result["predictions"].cpu())
                all_labels.append(result["labels"].cpu())
                seen += result["predictions"].shape[0]
    if not all_predictions:
        raise RuntimeError("Classifier eval produced no samples")
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    if predictions.shape != labels.shape or labels.ndim != 2 or labels.shape[1] != len(class_names):
        raise ValueError(
            f"Classifier predictions/labels {predictions.shape}/{labels.shape} "
            f"do not match data.deg_types={class_names}"
        )
    report = stage.compute_binary_metrics(predictions, labels)
    summary = {key: value for key, value in report.items() if key != "per_class"}
    report["class_names"] = class_names
    report["num_samples"] = labels.shape[0]
    report["per_class"] = [
        {"name": name, **values} for name, values in zip(class_names, report["per_class"], strict=True)
    ]
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
                batch = _limit_batch(
                    batch, None if num_samples_per_task is None else num_samples_per_task - seen
                )
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
                result = stage.eval_step(raw_model, raw_model, eval_batch)
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

                    if save_images and output_dir is not None and seen + index < images_per_task:
                        vis_paths.append(
                            _save_strip(
                                output_dir / "eval",
                                step,
                                str(batch_task_names[index]),
                                seen + index,
                                lq_crop[index],
                                pred,
                                target,
                                extra_panels={
                                    key: result[key][index, :, : pred.shape[-2], : pred.shape[-1]]
                                    for key in ("baseline", "vae_gt")
                                }
                                if "baseline" in result
                                else None,
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
            raw_model,
            loaders,
            cfg,
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


def predict_tiles(model, tensor, text_embedding, image_id, tile_size, overlap):
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= tile_overlap < tile_size")
    height, width = tensor.shape[-2:]
    if height <= tile_size and width <= tile_size:
        return model(tensor, text_embedding, noise_seeds=model.noise_seeds([image_id])).clamp(-1, 1)

    def starts(length):
        end = max(0, length - tile_size)
        return sorted(set([*range(0, end + 1, tile_size - overlap), end]))

    result = torch.zeros(tensor.shape, dtype=torch.float32, device="cpu")
    total = torch.zeros((1, 1, height, width), dtype=torch.float32)
    for top in starts(height):
        for left in starts(width):
            tile = tensor[..., top : top + tile_size, left : left + tile_size]
            seeds = model.noise_seeds([f"{image_id}:tile:{top}:{left}"])
            prediction = model(tile, text_embedding, noise_seeds=seeds).clamp(-1, 1).float().cpu()
            h, w = tile.shape[-2:]
            wy = torch.hann_window(h, periodic=False).clamp_min(1e-3)
            wx = torch.hann_window(w, periodic=False).clamp_min(1e-3)
            weight = wy[:, None] * wx[None, :]
            result[..., top : top + h, left : left + w] += prediction * weight
            total[..., top : top + h, left : left + w] += weight
    return result / total


def _inference_files(
    input_path: str | Path, gt_dir: str | Path | None
) -> tuple[list[tuple[Path, Path]], dict[str, Path]]:
    """Resolve all output names and GT matches before encoding or writing."""
    input_root = Path(input_path)
    files = []
    outputs = {}
    for image_path in list_input_images(input_path):
        relative = image_path.relative_to(input_root) if input_root.is_dir() else Path(image_path.name)
        output = relative.with_suffix(".png")
        if output in outputs:
            raise ValueError(f"Output path collision {output}: {outputs[output]} and {image_path}")
        outputs[output] = image_path
        files.append((image_path, output))
    gt_files = {}
    if gt_dir is not None:
        root = Path(gt_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"GT directory not found: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.stem in gt_files:
                raise ValueError(f"Ambiguous GT stem {path.stem!r}: {gt_files[path.stem]} and {path}")
            gt_files[path.stem] = path
        missing = [path for path, _ in files if path.stem not in gt_files]
        if missing:
            raise FileNotFoundError(f"No GT image matching stem {missing[0].stem} in {root}")
    return files, gt_files


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
    """Native input -> pad -> forward -> crop padding -> save.

    data.preprocessing.inference optionally transforms LQ/GT together before
    padding. Unlike dataset benchmarks, inference does not crop to a multiple.
    """
    if str(cfg.stage) != "spade":
        raise ValueError(f"run_inference is only available for stage=spade, got {cfg.stage}")
    files, gt_files = _inference_files(input_path, gt_dir)
    options = {"image_size": 0, **preprocessing_options(cfg, "inference", {})}
    preprocessing = PairedTransform(is_train=False, **options)
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
    saved_paths: list[Path] = []
    with torch.no_grad():
        for image_path, output_relative in tqdm(files, desc="Inference"):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if gt_dir is not None:
                with Image.open(gt_files[image_path.stem]) as source:
                    gt = source.convert("RGB")
                image, gt = preprocessing.geometry(image, gt)
            else:
                image, _ = preprocessing.geometry(image, image)
            width, height = image.size
            tensor = (
                torch.as_tensor(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            tensor = _pad_to_multiple(tensor, pad_multiple).to(device=device, dtype=weight_dtype)
            if cfg.eval.patchwise:
                tile_size, overlap = int(cfg.eval.tile_size), int(cfg.eval.tile_overlap)
                if tile_size % pad_multiple or overlap % pad_multiple:
                    raise ValueError("Tile size and overlap must be multiples of eval.pad_to_multiple")
                prediction = predict_tiles(
                    raw_model, tensor, text_embedding, str(image_path.resolve()), tile_size, overlap
                )
            else:
                prediction = raw_model(
                    tensor, text_embedding, noise_seeds=raw_model.noise_seeds([str(image_path.resolve())])
                ).clamp(-1, 1)
            prediction = prediction[:, :, :height, :width]
            prediction_np = metrics.to_numpy_rgb(prediction[0])
            output_path = save_dir / output_relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.clip(prediction_np, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output_path)
            saved_paths.append(output_path)

            if accumulator is not None:
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
    payload = {
        "step": report.step,
        "task_metrics": report.task_metrics,
        "overall": report.overall,
        "vis_paths": [str(path) for path in report.vis_paths],
    }
    if report.classification is not None:
        payload["classification"] = report.classification
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
