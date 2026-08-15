"""Pure metric helpers and the task-equal aggregation accumulator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

METRIC_NAMES = ("psnr", "ssim", "lpips")


def to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor (C,H,W) or (1,C,H,W) -> [0, 1] float32 HWC ndarray."""
    if image.dim() == 4:
        image = image[0]
    array = (image.detach().float().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(skimage_psnr(to_numpy_rgb(target), to_numpy_rgb(prediction), data_range=1.0))


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        skimage_ssim(
            to_numpy_rgb(target),
            to_numpy_rgb(prediction),
            data_range=1.0,
            channel_axis=-1,
        )
    )


def compute_lpips(lpips_fn: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        value = lpips_fn(prediction.float(), target.float())
    return float(torch.as_tensor(value).squeeze().mean().detach().cpu())


def load_lpips(net: str = "vgg", device: torch.device | None = None) -> torch.nn.Module:
    """Load LPIPS once; it is intentionally never a trainable checkpoint parameter."""
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "LPIPS is required when lpips metrics/loss are enabled. Install: pip install lpips"
        ) from exc
    model = lpips.LPIPS(net=net)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model


class MetricAccumulator:
    """Collect per-sample metrics, then report per-task means and task-equal overall means."""

    def __init__(self, metric_names: Sequence[str] = METRIC_NAMES) -> None:
        self.metric_names = tuple(metric_names)
        self._values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(self, task_name: str, values: dict[str, float]) -> None:
        for name in self.metric_names:
            if name in values and values[name] is not None:
                self._values[task_name][name].append(float(values[name]))

    def per_task(self) -> dict[str, dict[str, float]]:
        report: dict[str, dict[str, float]] = {}
        for task_name in sorted(self._values):
            report[task_name] = {
                name: float(np.mean(self._values[task_name][name]))
                for name in self.metric_names
                if self._values[task_name][name]
            }
        return report

    def overall(self) -> dict[str, float]:
        """Task-equal overall: average of per-task means (not sample-weighted)."""
        per_task = self.per_task()
        report: dict[str, float] = {}
        for name in self.metric_names:
            values = [metrics[name] for metrics in per_task.values() if name in metrics]
            if values:
                report[name] = float(np.mean(values))
        return report
