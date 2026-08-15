#!/usr/bin/env python3
"""Single evaluation / inference entry point.

Benchmark (uses config test tasks + GT)::

    accelerate launch eval.py --config configs/stage3_spade.yaml \
        --checkpoint output/stage3/final/weights.safetensors --use_ema

Inference (pad -> shared forward -> crop back -> save)::

    python eval.py --config configs/stage3_spade.yaml \
        --checkpoint output/stage3/final/weights.safetensors \
        --input /path/to/lq_images --save_dir output/inference \
        --prompt "a high quality clean image"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from accelerate import Accelerator

from sd_aio import checkpoint, metrics, utils
from sd_aio import config as configlib
from sd_aio import data as datalib
from sd_aio.eval import run_eval, run_inference, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or run inference with an SD-AiO checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="safetensors file, checkpoint-* dir, or output_dir",
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--input", default=None, help="single image or image directory for inference")
    parser.add_argument("--gt", default=None, help="optional GT directory used with --input for metrics")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    return parser.parse_args(argv)


def load_stage(stage_name: str) -> Any:
    import importlib

    return importlib.import_module(f"sd_aio.{stage_name}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = configlib.load_config(args.config, args.overrides)
    if args.tile_size is not None:
        cfg.eval.tile_size = args.tile_size
    stage = load_stage(str(cfg.stage))

    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    device = accelerator.device
    model = stage.build_model(cfg, device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.output_dir)
    weights_path = checkpoint.resolve_weights_path(checkpoint_path, prefer_ema=args.use_ema)
    if accelerator.is_main_process:
        print(f"Loading weights from {weights_path}")
    checkpoint.load_model_weights(model, weights_path)

    model = accelerator.prepare(model)
    raw_model = accelerator.unwrap_model(model)
    weight_dtype = utils.weight_dtype_for(str(cfg.mixed_precision))
    stage.set_eval_mode(raw_model)

    if args.input is not None:
        save_dir = Path(args.save_dir or (Path(cfg.output_dir) / "inference"))
        paths, report = run_inference(
            stage,
            model,
            raw_model,
            cfg,
            args.input,
            save_dir,
            device=device,
            weight_dtype=weight_dtype,
            prompt=args.prompt,
            gt_dir=args.gt,
        )
        if accelerator.is_main_process:
            print(f"Saved {len(paths)} image(s) to {save_dir}")
            if report is not None:
                print(report.text())
                save_report(report, save_dir / "metrics.json")
    else:
        _, test_loaders = datalib.build_loaders(cfg, verbose=accelerator.is_main_process)
        if not test_loaders:
            raise RuntimeError("No test tasks in config and no --input provided")
        eval_lpips = None
        if bool(cfg.eval.get("compute_lpips", True)) and str(cfg.stage) != "classifier":
            if getattr(raw_model, "lpips", None) is not None:
                eval_lpips = raw_model.lpips
            else:
                eval_lpips = metrics.load_lpips(str(cfg.eval.get("lpips_net", "vgg")), device)
        report = run_eval(
            stage,
            model,
            raw_model,
            test_loaders,
            cfg,
            device=device,
            weight_dtype=weight_dtype,
            lpips_fn=eval_lpips,
            step=0,
            output_dir=Path(cfg.output_dir),
            save_images=True,
            num_samples_per_task=args.num_samples,
        )
        if accelerator.is_main_process:
            print(report.text())
            save_report(report, Path(cfg.output_dir) / "eval" / "metrics_standalone.json")
    accelerator.end_training()


if __name__ == "__main__":
    main()
