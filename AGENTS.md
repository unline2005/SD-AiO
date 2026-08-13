# SD-AiO — Single-Step All-in-One Degradation Restoration

## Current goal

Build a **single-step all-in-one image restoration** system on Stable Diffusion 2.1 for 3D degradations:

- Dehaze
- Derain
- Gaussian denoise at σ=15/25/50

The current research question is no longer “can frozen SD + SFT run”; it is whether the current SD-AiO baseline can be pushed toward DOD-style one-step restoration quality by fixing infrastructure, then moving toward UNet-wide degradation-aware LoRA / CLoRA.

## Architecture

```text
LQ Image
  → VAE Encoder
  → z_LQ + fixed-timestep noise
  → UNet with condition module hooks
  → z_hat
  → VAE Decoder
  → Restored image

LQ Image
  → DINOv2 degradation classifier
  → DegFeatExtractor
  → F_Deg
  → condition module
  → UNet modulation
```

## Current condition modules

Implemented registry keys in `src/cond_module.py`:

| Key | Name | Status | Core idea |
|---|---|---|---|
| `none` | Identity | Utility | No conditioning |
| `codsr_lqfm` | SimpleSFTModule | Legacy / 1b-ish | LQ CNN feature pyramid → SFT heads |
| `deg_aware_sft` | DegAwareSFTModule | 1c | LQ CNN features gated by F_Deg → SFT heads hooked on conv outputs |
| `deg_resblock_attn` | ResBlockAttnModule | 1a | F_Deg → degradation tokens → CrossAttn inside ResBlocks |

Important: `deg_cross_attn` is a stale historical name and is **not valid**. Scripts must use `deg_resblock_attn` for 1a.

## F_Deg current design

`src/degnet.py` uses Scheme B:

```text
F_Deg = cls_token + deg_alpha * (probs @ deg_embedding)
```

- `cls_token`: continuous DINOv2 [CLS] representation.
- `probs @ deg_embedding`: explicit degradation-type prototype residual.
- `deg_alpha`: trainable scalar initialized to 10.0.
- DINOv2 classifier is frozen during stage 2.

## Data protocol

Main 3D config: `configs/tasks_3d.yaml`.

Current train/test split:

| Degradation | Train | Test | Notes |
|---|---|---|---|
| Dehaze | OTS_BETA haze/clear | SOTS outdoor | prefix match by first 4 chars |
| Derain | RainTrainL/rain + RainTrainL/norain | Rain100L/rainy + norain | RainTrainL is train; Rain100L is test |
| Denoise | BSD400 + WaterlooED clean images | BSD68 clean images | online Gaussian noise |

Current repeat ratios:

- Dehaze: ×1 → 72,135
- Derain: ×200 → 40,000
- BSD400: ×100 → 40,000
- WaterlooED: ×2 → ~9,492

Training uses a single unified shuffled pool in `src/utils/dataset.py` via `build_unified_train_dataset`.

Known fixed data bug: `_add_noise()` must treat PIL tensors as `[0,1]` and multiply by 255 before adding σ-scale noise. Current code does this correctly.

## Evaluation protocol

Current integrated eval in `src/train.py`:

- RGB PSNR / RGB SSIM / LPIPS.
- `data_range=1`.
- Eval images are center-cropped to multiples of 16 where configured, then padded to multiples of 64 for SD forward and cropped back.
- Denoise eval adds online noise with deterministic crc32 seeds.
- Overall metric is task-equal: mean each task first, then average tasks.

Do not assume Y-channel metrics unless the current eval code explicitly uses them.

## Current scripts

Key scripts under `scripts/`:

| Script | Purpose |
|---|---|
| `train_1a_t100_2gpu.sh` | 1a / `deg_resblock_attn`, t=100, VAE LoRA=16, 2 GPUs |
| `train_1c_t100_2gpu.sh` | 1c / `deg_aware_sft`, t=100, VAE LoRA=16, 2 GPUs |
| `train_1c_t100_1gpu.sh` | 1c, 1 GPU |
| `train_1a_t200_1gpu.sh` | older 1a/t=200 script |
| `train_classifier_3d.sh` | stage-1 classifier for 3D setup |
| `reeval_ckpt.py` | standalone reevaluation; verify timestep/condition type before trusting results |

Root-level old scripts may be deleted/stale. Prefer `scripts/`.

## Current empirical state

1c with L1=2.0 + LPIPS=5.0 plateaued around:

- Overall PSNR ≈ 24.87
- SSIM ≈ 0.712
- LPIPS ≈ 0.262

Derain was worse than LQ baseline:

- Rain100L LQ baseline ≈ 25.5 dB
- Model derain ≈ 22.x dB

Interpretation: for subtle rain degradation, the current one-step SD path over-edits the image. LPIPS=5.0 and t=100 are especially harmful for PSNR on light degradations.

## Current known bottlenecks

Do not present these as newly discovered bugs:

- UNet mostly frozen: high-level method limitation.
- No DOD-style CLoRA / UNet-wide degradation-aware LoRA yet.
- No HDE / decoder RRDB detail enhancement yet.
- No DMD distillation yet.
- LPIPS=5.0 is a deliberate but likely bad PSNR-oriented hyperparameter, not a code bug.

## DOD comparison context

DOD’s reported quality mainly comes from:

- UNet-wide degradation-aware LoRA / CLoRA.
- Decoder detail enhancement / HDE.
- DMD distillation.
- Two-stage training.

The public DOD repo infrastructure is not perfectly aligned with the paper protocol: it hardcodes many paths/ratios, has custom multi-task batch construction, and does not publish a full paper-table evaluator. When comparing, first align dataset split, denoise protocol, timestep, loss, and metric definitions.

## Current likely next directions

Priority order:

1. Fix any concrete infrastructure bugs that silently corrupt eval/training.
2. Run or inspect 1a (`deg_resblock_attn`) under the corrected 3D data protocol.
3. Stop using LPIPS=5.0 for PSNR-oriented runs; test LPIPS=0.1/0.2 or per-task LPIPS.
4. Add UNet LoRA/CLoRA-style degradation-aware adaptation, preferably by studying/migrating DOD infrastructure.
5. Consider decoder detail enhancement only after the core train/eval protocol is clean.

## Style / implementation preferences

- Native PyTorch + `nn.Module`; avoid complex inheritance and framework-style overengineering.
- Prefer simple, inspectable code paths.
- Cite exact `file:line` when discussing code.
- Be strict about distinguishing code/protocol bugs from research-method limitations.
- Before claiming a metric gap is architectural, first check data pairing, metric definition, timestep, crop, loss weights, checkpoint loading, and script/config drift.
