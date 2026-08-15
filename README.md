# SD-AiO

Clean, config-driven re-implementation of single-step SD 2.1 all-in-one image
restoration (haze / rain / Gaussian denoise). One training entry point, one
evaluation/inference entry point, and every stage is a self-contained module.

## Design principles

- **One loop, three stages.** `train.py` contains the only training loop.
  Adding a stage means adding one file under `sd_aio/`.
- **Config is the experiment.** `OmegaConf` YAML + `key=value` CLI overrides.
  No argparse/script drift, no silent defaults.
- **One forward for train and inference.** `SpadeRestorer.forward` is used by
  training, periodic eval, standalone eval and `--input` inference.
- **Frozen things stay frozen.** The train-mode helper never switches frozen
  GroupNorm back to batch statistics.
- **Checkpoints are safe and small.** Only trainable weights are saved, as
  `safetensors`; `strict` loading catches config/checkpoint mismatch.

## Layout

```text
train.py / eval.py          # the only two entry points
sd_aio/
├── config.py               # defaults merge + strict get + path resolution
├── data.py                 # pairing / online noise / transforms / loaders
├── eval.py                 # run_eval() + run_inference() (shared eval core)
├── metrics.py              # PSNR / SSIM / LPIPS + task-equal overall
├── checkpoint.py           # safetensors checkpoints, resume, EMA
├── classifier.py           # Stage 1: DINOv2 classifier + F_Deg extractor
├── vae_encoder.py          # Stage 2: PreRestoreEncoder + AdaIN latent alignment
├── spade.py                # Stage 3: SpadeRestorer (one-step SD restoration)
└── utils.py                # train/eval mode helpers, dtype helpers
configs/
├── defaults.yaml           # the only implicit defaults
├── tasks_3d.yaml           # shared dataset definition (server paths)
└── stage{1,2,3}_*.yaml     # one file per stage / experiment
tests/                      # real pytest regression tests (no grep tests)
```

## Install

```bash
pip install -r requirements.txt
```

## Quickstart (yhmi server paths)

All dataset paths live in `configs/tasks_3d.yaml`; all model paths live in the
stage YAMLs.

```bash
# Stage 1: degradation classifier
accelerate launch --num_processes=1 --mixed_precision=bf16 train.py \
    --config configs/stage1_classifier.yaml

# Stage 2: VAE encoder latent alignment
accelerate launch --num_processes=1 --mixed_precision=bf16 train.py \
    --config configs/stage2_vae.yaml

# Stage 3: single-step restoration
accelerate launch --num_processes=2 --mixed_precision=bf16 train.py \
    --config configs/stage3_spade.yaml
```

Override anything without touching YAML:

```bash
accelerate launch train.py --config configs/stage3_spade.yaml \
    trainer.max_steps=1000 model.lora.unet_rank=8
```

## Stage protocol

Every stage module exports the same six functions. `model` is the
accelerator-prepared model (DDP/autocast aware); `raw_model` is its unwrapped
counterpart used for attribute access only.

| Function | Responsibility |
|---|---|
| `build_model(cfg, device)` | Build the stage model; frozen submodules are frozen at build time |
| `make_optimizer(model, cfg)` | Build optimizer, optionally with parameter groups |
| `set_train_mode(model)` | Train mode for trainable leaves only |
| `set_eval_mode(model)` | Eval mode |
| `compute_loss(model, raw_model, batch, cfg)` | One loss step, returns `(loss, logs)` |
| `eval_step(model, raw_model, batch, cfg)` | One no-grad eval step |

## Evaluation and inference

```bash
# Full benchmark from config test tasks
python eval.py --config configs/stage3_spade.yaml \
    --checkpoint output/stage3_spade/final/weights.safetensors --use_ema

# Single image or directory; pad -> shared forward -> crop back
python eval.py --config configs/stage3_spade.yaml \
    --checkpoint output/stage3_spade/final/weights.safetensors \
    --input /path/to/lq_images --save_dir output/inference \
    --prompt "a high quality clean image"
```

Eval protocol: center-crop to a multiple of `eval.crop_to_multiple` (16),
reflect-pad to a multiple of `eval.pad_to_multiple` (64), forward, crop back.
Metrics are per-task means aggregated task-equally. VAE tiling is opt-in with
`eval.tiling` and uses diffusers' native tiled encode/decode.

## Checkpoints

```text
output_dir/
├── config.yaml                  # build snapshot: eval can never drift
├── train.log / metrics.jsonl
├── checkpoints/
│   └── checkpoint-00001000/
│       ├── weights.safetensors  # trainable parameters only
│       ├── ema.safetensors      # optional
│       ├── optimizer.pt
│       └── state.json
└── final/weights.safetensors    # publication entry point
```

Resume with `trainer.resume_from: latest` or a checkpoint path. A corrupt
latest checkpoint is skipped automatically in favour of the previous one.

## Tests

```bash
pytest          # 34 CPU regression tests, including a one-step end-to-end run
ruff check train.py eval.py sd_aio tests
```

The tests build tiny SD-shaped components locally; they do not download model
weights.

## Notes for the server

- `/data` is the data root; model roots are `/root/shared-nvme/model/...`.
- Stage 2 and Stage 3 expect the Stage 1 classifier at the path in their YAML.
- For 8 GPUs use `accelerate launch --multi_gpu --num_processes=8`.
- Logging is main-process-only; periodic eval runs on the main process and is
  followed by an `accelerator.wait_for_everyone()` barrier.
