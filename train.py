from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from sd_aio import checkpoint, metrics, runtime, utils
from sd_aio import config as configlib
from sd_aio import data as datalib
from sd_aio.eval import run_eval, save_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SD-AiO stage")
    parser.add_argument("--config", required=True, help="Stage YAML config")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE", help="OmegaConf dot-path overrides")
    return parser.parse_args(argv)


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
    configlib.validate_training(cfg)
    stage_name = str(cfg.stage)
    stage = utils.load_stage(stage_name)

    log_with = cfg.trainer.get("log_with")
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=int(cfg.trainer.distributed_timeout_seconds)))
        ]
        + ([DistributedDataParallelKwargs(broadcast_buffers=False)] if stage_name == "classifier" else []),
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        mixed_precision=str(cfg.mixed_precision),
        log_with=log_with,
        # Classifier max_steps counts optimizer updates, independent of world size.
        step_scheduler_with_optimizer=False,
    )
    is_main = accelerator.is_main_process
    is_local_main = accelerator.is_local_main_process
    if cfg.seed is not None:
        set_seed(int(cfg.seed))

    resume_path = _resolve_resume_path(cfg)
    if resume_path is not None and hasattr(stage, "validate_resume"):
        stage.validate_resume(cfg, resume_path)
    output_dir = Path(cfg.output_dir)
    logger = setup_logger(output_dir, is_main)
    if is_main:
        configlib.snapshot(cfg, output_dir)
        runtime.write_metadata(output_dir, world_size=accelerator.num_processes, seed=cfg.seed)
        logger.info("SD-AiO training | stage=%s | output=%s", stage_name, output_dir)
        logger.info("Config: %s", args.config)

    train_loader, test_loaders = datalib.build_loaders(
        cfg, verbose=is_main, eval_split=str(cfg.trainer.get("eval_split", "test"))
    )
    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty; check cfg.data.train paths and batch size")
    model = stage.build_model(cfg, accelerator.device)

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
        global_step, restored_dir = checkpoint.restore_training(raw_model, optimizer, scheduler, resume_path)
        if is_main:
            logger.info("Resumed from %s at step %d", restored_dir, global_step)

    ema = None
    if bool(cfg.ema.enabled):
        ema = checkpoint.ModelEMA(raw_model, decay=float(cfg.ema.decay))
        if resume_path is not None:
            ema_file = Path(restored_dir) / checkpoint.EMA_NAME
            if not ema_file.exists():
                raise FileNotFoundError(f"EMA resume requires {ema_file}")
            ema.load_state_dict(ema_file)

    weight_dtype = utils.weight_dtype_for(str(cfg.mixed_precision))
    eval_lpips = None
    if bool(cfg.eval.get("compute_lpips", True)) and stage_name != "classifier" and test_loaders and is_main:
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
    completed = False
    metric_window = runtime.MetricWindow()

    try:
        while global_step < max_steps:
            for batch in train_loader:
                batch = utils.move_batch(batch, accelerator.device, weight_dtype)
                with accelerator.accumulate(model):
                    loss, logs = stage.compute_loss(model, raw_model, batch, cfg)
                    metric_window.add({**logs, "loss": loss.detach()})
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if hasattr(stage, "before_optimizer_step"):
                            stage.before_optimizer_step(raw_model, cfg, global_step)
                        accelerator.clip_grad_norm_(model.parameters(), float(cfg.trainer.max_grad_norm))
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue
                logs = metric_window.pop()
                if accelerator.optimizer_step_was_skipped:
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
                    if is_main:
                        runtime.append_metrics(
                            output_dir,
                            global_step,
                            {"loss": gathered, "lr": lr, "seconds_per_step": 1.0 / max(steps_per_sec, 1e-12)},
                        )
                    if "loss_pixel_mse" in logs or "loss_pixel_l1" in logs:
                        pixel_key = "loss_pixel_l1" if "loss_pixel_l1" in logs else "loss_pixel_mse"
                        components = (
                            accelerator.gather(
                                torch.tensor(
                                    [logs[pixel_key], logs.get("loss_lpips", 0.0)], device=accelerator.device
                                )
                            )
                            .reshape(-1, 2)
                            .mean(0)
                        )
                        if is_main:
                            logger.info(
                                "loss terms (unweighted) | %s=%.6f | lpips=%.6f",
                                pixel_key,
                                *components.tolist(),
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
                        tsne_cfg = cfg.eval.get("tsne")
                        if (
                            stage_name == "classifier"
                            and tsne_cfg is not None
                            and bool(tsne_cfg.enabled)
                            and global_step % int(tsne_cfg.every_steps) == 0
                        ):
                            tsne_dir = stage.save_tsne_visualization(
                                raw_model,
                                test_loaders,
                                cfg,
                                device=accelerator.device,
                                weight_dtype=weight_dtype,
                                output_dir=output_dir,
                                step=global_step,
                            )
                            logger.info("Saved classifier t-SNE to %s", tsne_dir)
                    accelerator.wait_for_everyone()
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
        completed = True

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
            if completed and test_loaders:
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
            if completed:
                checkpoint.save_final(raw_model, output_dir, ema=ema)
                logger.info(
                    "Training finished at step %d; final weights in %s/final",
                    finished_step,
                    output_dir,
                )
        accelerator.end_training()


if __name__ == "__main__":
    main()
