#!/usr/bin/env python3
"""Single training entry point for every stage.

Usage::

    accelerate launch train.py --config configs/stage1_classifier.yaml
    accelerate launch train.py --config configs/stage3_spade.yaml trainer.max_steps=1000

The stage is chosen by ``stage:`` in the YAML file.  This file contains the
only training loop in the repository; stage modules only implement the
six-function protocol (build / optimizer / train-mode / eval-mode / loss /
eval-step).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from sd_aio import checkpoint, metrics, utils
from sd_aio import config as configlib
from sd_aio import data as datalib
from sd_aio.eval import run_eval, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SD-AiO stage")
    parser.add_argument("--config", required=True, help="Stage YAML config")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE", help="OmegaConf dot-path overrides")
    return parser.parse_args(argv)


def load_stage(stage_name: str) -> Any:
    module_name = f"sd_aio.{stage_name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"Unknown stage {stage_name}; expected a module sd_aio/{stage_name}.py"
        ) from exc


def setup_logger(output_dir: Path, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("sd_aio.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _resolve_resume_path(cfg: OmegaConf) -> Path | None:
    resume_from = cfg.trainer.get("resume_from")
    if resume_from is None:
        return None
    resume_from = str(resume_from)
    if resume_from.lower() == "latest":
        latest = checkpoint.find_latest_checkpoint(cfg.output_dir)
        if latest is None:
            raise FileNotFoundError(f"No latest checkpoint found under {cfg.output_dir}")
        return latest
    return Path(resume_from)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = configlib.load_config(args.config, args.overrides)
    stage_name = str(cfg.stage)
    stage = load_stage(stage_name)

    log_with = cfg.trainer.get("log_with")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        mixed_precision=str(cfg.mixed_precision),
        log_with=log_with,
    )
    is_main = accelerator.is_main_process
    is_local_main = accelerator.is_local_main_process
    if cfg.seed is not None:
        set_seed(int(cfg.seed))

    output_dir = Path(cfg.output_dir)
    logger = setup_logger(output_dir, is_main)
    if is_main:
        configlib.snapshot(cfg, output_dir)
        logger.info("SD-AiO training | stage=%s | output=%s", stage_name, output_dir)
        logger.info("Config: %s", args.config)

    model = stage.build_model(cfg, accelerator.device)
    train_loader, test_loaders = datalib.build_loaders(cfg, verbose=is_main)
    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("Train loader is empty; check cfg.data.train paths and batch size")

    optimizer = stage.make_optimizer(model, cfg)
    scheduler = get_scheduler(
        str(cfg.scheduler.name),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.scheduler.warmup_steps),
        num_training_steps=int(cfg.trainer.max_steps),
    )
    if is_main:
        total = utils.count_parameters(model, trainable_only=False)
        trainable = utils.count_parameters(model, trainable_only=True)
        logger.info("Parameters: total=%.2fM trainable=%.2fM", total / 1e6, trainable / 1e6)

    model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)
    raw_model = accelerator.unwrap_model(model)

    global_step = 0
    resume_path = _resolve_resume_path(cfg)
    if resume_path is not None:
        global_step, restored_dir = checkpoint.restore_training(
            raw_model,
            optimizer,
            scheduler,
            resume_path,
            prefer_ema=bool(cfg.trainer.get("resume_use_ema", False)),
        )
        if is_main:
            logger.info("Resumed from %s at step %d", restored_dir, global_step)

    ema = None
    if bool(cfg.ema.enabled):
        ema = checkpoint.ModelEMA(raw_model, decay=float(cfg.ema.decay))
        if resume_path is not None:
            ema_file = Path(restored_dir) / checkpoint.EMA_NAME
            if ema_file.exists():
                ema.load_state_dict(ema_file)

    weight_dtype = utils.weight_dtype_for(str(cfg.mixed_precision))
    eval_lpips = None
    if bool(cfg.eval.get("compute_lpips", True)) and stage_name != "classifier":
        if getattr(raw_model, "lpips", None) is not None:
            eval_lpips = raw_model.lpips
        else:
            eval_lpips = metrics.load_lpips(str(cfg.eval.get("lpips_net", "vgg")), accelerator.device)

    stage.set_train_mode(raw_model)
    max_steps = int(cfg.trainer.max_steps)
    log_every = int(cfg.trainer.log_every)
    eval_freq = int(cfg.trainer.eval_freq)
    checkpoint_steps = int(cfg.trainer.checkpointing_steps)
    progress = tqdm(
        range(max_steps),
        initial=global_step,
        desc=stage_name,
        disable=not is_local_main,
    )

    loss_value = 0.0
    start_time = time.time()
    finished_step = global_step

    try:
        for _epoch in range(1_000_000):
            for batch in train_loader:
                batch = utils.move_batch(batch, accelerator.device, weight_dtype)
                with accelerator.accumulate(model):
                    loss, logs = stage.compute_loss(model, raw_model, batch, cfg)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), float(cfg.trainer.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                finished_step = global_step
                progress.update(1)
                loss_value = float(logs.get("loss", loss.detach()))
                progress.set_postfix(loss=f"{loss_value:.4f}")

                if ema is not None:
                    ema.update(raw_model)

                if log_every > 0 and global_step % log_every == 0:
                    gathered = (
                        accelerator.gather(torch.tensor([loss_value], device=accelerator.device))
                        .mean()
                        .item()
                    )
                    lr = float(scheduler.get_last_lr()[0])
                    elapsed = time.time() - start_time
                    steps_per_sec = log_every / max(elapsed, 1e-6)
                    if is_main:
                        logger.info(
                            "step %d/%d | loss=%.4f | lr=%.2e | %.2fs/it",
                            global_step,
                            max_steps,
                            gathered,
                            lr,
                            1.0 / max(steps_per_sec, 1e-12),
                        )
                    start_time = time.time()

                if eval_freq > 0 and global_step % eval_freq == 0 and test_loaders:
                    if is_main:
                        report = run_eval(
                            stage,
                            model,
                            raw_model,
                            test_loaders,
                            cfg,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            lpips_fn=eval_lpips,
                            step=global_step,
                            output_dir=output_dir,
                            save_images=True,
                            num_samples_per_task=cfg.trainer.get("eval_num_samples"),
                        )
                        logger.info(report.text())
                        save_report(
                            report,
                            output_dir / "eval" / f"metrics_step_{global_step:08d}.json",
                        )
                        with (output_dir / "metrics.jsonl").open("a") as metrics_file:
                            metrics_file.write(
                                json.dumps(
                                    {
                                        "step": global_step,
                                        "task_metrics": report.task_metrics,
                                        "overall": report.overall,
                                    }
                                )
                                + "\n"
                            )
                    accelerator.wait_for_everyone()
                    if is_main:
                        stage.set_train_mode(raw_model)

                if is_main and checkpoint_steps > 0 and global_step % checkpoint_steps == 0:
                    checkpoint.save_checkpoint(
                        raw_model,
                        optimizer,
                        scheduler,
                        global_step,
                        output_dir,
                        ema=ema,
                        keep_last=int(cfg.keep_last_checkpoints),
                    )
                    logger.info("Saved checkpoint at step %d", global_step)

                if global_step >= max_steps:
                    break
            if global_step >= max_steps:
                break
    finally:
        accelerator.wait_for_everyone()
        if is_main:
            if checkpoint_steps > 0:
                checkpoint.save_checkpoint(
                    raw_model,
                    optimizer,
                    scheduler,
                    finished_step,
                    output_dir,
                    ema=ema,
                    keep_last=int(cfg.keep_last_checkpoints),
                )
            if test_loaders:
                report = run_eval(
                    stage,
                    model,
                    raw_model,
                    test_loaders,
                    cfg,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    lpips_fn=eval_lpips,
                    step=finished_step,
                    output_dir=output_dir,
                    save_images=True,
                )
                logger.info("Final eval: %s", report.text())
                save_report(
                    report,
                    output_dir / "eval" / f"metrics_step_{finished_step:08d}.json",
                )
            checkpoint.save_final(raw_model, output_dir, ema=ema)
            logger.info(
                "Training finished at step %d; final weights in %s/final",
                finished_step,
                output_dir,
            )
        accelerator.end_training()


if __name__ == "__main__":
    main()
