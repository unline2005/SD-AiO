# AGENTS.md

Guidance for coding agents working on this repository.

## Project shape

- `train.py` and `eval.py` are the only entry points.
- The training loop exists exactly once, in `train.py`.
- Each stage (`sd_aio/classifier.py`, `sd_aio/vae_encoder.py`,
  `sd_aio/spade.py`) is self-contained and exports the six-function stage
  protocol documented in `README.md`.
- `sd_aio/eval.py` is the only evaluation path; training-periodic eval,
  standalone benchmark and inference all go through it.

## Rules

1. **No new argparse options for experiments.** Add YAML keys and use
   `configs/defaults.yaml` for shared defaults. CLI is only for
   `--config`, `key=value` overrides, and the fixed eval/inference switches.
2. **No silent config defaults in code.** Use `config.required()` or a visible
   key in `defaults.yaml`.
3. **Never call `model.train()` on a whole multi-component model.** Use
   `utils.set_train_mode` / `utils.set_eval_mode` so frozen VAE/DINO/GroupNorm
   submodules stay in eval mode.
4. **Prepared vs raw model.** In distributed training `model` may be a DDP
   wrapper. Stage functions receive both `model` (use for forward/backward)
   and `raw_model` (use for attributes, caches and saving).
5. **Checkpoints contain trainable parameters only, as safetensors.** Rebuild
   is `build_model(config.yaml) + load_model_weights()`. Never add
   `strict=False` loads that hide mismatches.
6. **Data errors must fail fast.** Missing dirs, zero images, pairing
   mismatch, denoise tasks without a `_<sigma>` suffix: raise, do not skip.
7. **Denoise is online and deterministic at eval time.**
   `sigma / 127.5` in [-1, 1] space; eval noise uses a crc32 seed.
8. **Text encoder stays on CPU.** `SpadeRestorer` keeps it in `_aux`
   (unregistered) after prompt caching; never register it as a submodule.
9. **Keep `tests/` real.** Every new numerical/data/checkpoint behaviour gets
   a CPU test. Run `pytest` and `ruff check train.py eval.py sd_aio tests`
   before finishing.
10. **README/AGENTS stay in sync with the code.** If you change the protocol,
    update both.

## Typical edit cycle

```bash
# 1. edit YAML / stage module
# 2. verify
python -m pytest -q
ruff check train.py eval.py sd_aio tests
ruff format train.py eval.py sd_aio tests
# 3. commit
git add -A && git commit -m "feat: ..."
```
