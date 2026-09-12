# Configuration, training and reproducibility

## Configuration composition

`defaults.yaml` provides visible shared defaults. Experiments can inherit one or
multiple YAML files with `_base_`; base paths resolve relative to the declaring
file. Later bases and the child override earlier values. CLI overrides take final
precedence. Cyclic inheritance and invalid training parameters fail before model
construction.

```yaml
_base_: stage2_vae_foundir_ggt_cdd11.yaml
output_dir: /home/yhmi/data/output/my_encoder_experiment
model:
  vae_training:
    encoder_init: random
    freeze_encoder: false
trainer:
  train_batch_size: 3
  gradient_accumulation_steps: 2
```

Save this example in `configs/`. Configuration snapshots embed the resolved task
definitions for all stages; later edits to a tasks file do not alter a saved run.
Environment variables can be referenced with OmegaConf's `${oc.env:NAME}` syntax.

## Existing strategies

| Concern | Implemented choices |
|---|---|
| Optimizer | `optimizer.name: adamw`, `adam`, `sgd`; native parameter groups |
| Scheduler | Diffusers `get_scheduler` strategies, e.g. cosine/linear/constant with warmup |
| VAE pixel objective | `loss.pixel_type: l1`, `mse`, `charbonnier` |
| Charbonnier | `loss.charbonnier_epsilon`, default `0.001` |
| Perceptual objective | `loss.lambda_lpips` and `loss.lpips_net`; disabled at zero weight |
| VAE supervision | Original GT, frozen original-VAE reconstruction, or legacy latent target |
| Encoder initialization | `pretrained` or `random`; decoder and quant convolutions remain frozen |
| Classifier | CLS MLP or label-query head; full or partial backbone fine-tuning |
| Classifier patch auxiliary target | Disabled, patch mean absolute difference, damaged-pixel fraction, or RGB residual |
| Restoration condition | Legacy global condition or query evidence |
| Memory/precision | Gradient accumulation, fp32/fp16/bf16, explicit full-pipeline tiles |

Stage3 retains its existing RGB MSE + LPIPS objective. Do not assume a VAE-only
loss option changes Stage3. Keep architecture-affecting choices in `model`, data
semantics in `data`, and objectives in `loss`; Stage2/3 resume validates them.

Stage1 patch experiments use one-based DINO block indices. Intermediate patch
tokens are normalized per layer, concatenated, reduced by a 1x1 convolution and
mixed locally by one 3x3 convolution. The auxiliary target requires aligned LQ/GT
pairs and `model.classifier.pad_to_patch_multiple: true`; padded pixels are
excluded from the target statistics. The auxiliary head is a training constraint,
not a physical degradation estimator.

## Saved state

- `config.yaml`: resolved experiment and dataset definitions.
- `runtime.json` / `runtime_history.jsonl`: software versions, world size, seed and code hashes.
- `metrics.jsonl`: optimizer-update loss averaged across accumulation and ranks, plus LR and timing.
- `checkpoints/checkpoint-N/`: trainable safetensors, frozen condition sidecar and optimizer/scheduler state.
- `final/`: weights after a normally completed training loop.

New checkpoints are assembled in hidden staging directories before publication.
Completion metadata covers auxiliary files. Readers continue to accept valid
legacy checkpoints and reject incomplete new saves. Weight keys and shapes are
checked before modifying the model.

`resume_from=latest` restores optimization progress; it does not guarantee exact
DataLoader/RNG replay of the same next minibatch. Switching optimizers while
resuming is not a supported experiment design.

## Validation, EMA and stopping

Training validation evaluates the ordinary model. EMA is saved separately and
can be evaluated explicitly with `eval.py --use_ema`. EMA resume requires saved
EMA state; frozen conditions are saved alongside new EMA weights.

`train.py` itself runs to `max_steps`; existing `trainer.early_stopping` metadata
is consumed by the external experiment supervisor. The random-encoder supervisor
selects the best validation checkpoint and starts Stage3 after its plateau rule.
Do not claim native early stopping or native best-checkpoint selection from a
standalone `train.py` invocation.

Only task-equal benchmark aggregation is currently implemented inside the shared
evaluator. Source-balanced selection is computed explicitly by the supervisors.
PSNR/SSIM use RGB in `[0,1]`; document any crop, color-space or noise-protocol
differences before comparing with published tables.
