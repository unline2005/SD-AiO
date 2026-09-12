# Working on SD-AiO

The method is work in progress. Keep research choices explicit and do not claim
model-family support without an implemented, tested pipeline.

## Architecture

- `train.py` is the only training loop; `eval.py` is the evaluation CLI.
- `sd_aio/eval.py` is the shared benchmark/inference path.
- Each stage is a normal module exporting `build_model`, `make_optimizer`,
  `set_train_mode`, `set_eval_mode`, `compute_loss`, and `eval_step`.
- Prefer `nn.Module` composition and small functions to inheritance trees,
  registries and duplicated loops. Add abstractions only for real consumers.

## Configuration and data

- Experiments belong in YAML. Shared defaults belong in `configs/defaults.yaml`;
  do not add experimental argparse flags or silent code defaults.
- `_base_` paths resolve relative to their declaring YAML. CLI overrides win.
- Snapshots embed tasks. Use `config.resume_section` for semantic resume checks.
- Model inputs are RGB tensors in `[-1,1]`. Paired geometry must stay aligned.
- Missing/ambiguous pairs, empty tasks, invalid labels and inconsistent sizes
  must fail explicitly. Do not silently skip data.
- Denoise uses `sigma/127.5`; evaluation noise is deterministic. Preserve the
  selected image/noise/metric protocol when reproducing an experiment.

## Training and persistence

- Stage functions receive prepared `model` and unwrapped `raw_model`. Use the
  former for trainable forward/backward, the latter for attributes and saving.
- Do not call `train()` on an entire mixed frozen/trainable model. Use stage
  mode helpers. GroupNorm has no running statistics; frozen Dropout/BatchNorm
  still require correct modes.
- Optimizer/scheduler steps count successful synchronized updates. Preserve
  effective batch semantics when changing device count or accumulation.
- Checkpoints contain trainable parameters only. Rebuild frozen pretrained
  modules from configuration and restore condition sidecars explicitly.
- Validate all checkpoint keys/shapes before loading. Publish completed saves;
  do not weaken mismatch checks with an unchecked `strict=False` load.
- Keep text encoders on CPU after prompt caching; `SpadeRestorer._aux` must not
  become a registered GPU submodule.
- Current early stopping/best selection belongs to external supervisors.
  Training validation uses ordinary weights; EMA evaluation is explicit.
- Resume restores optimization state, not exact DataLoader/RNG replay.

## Validation

Every new numerical, data or checkpoint behavior needs a meaningful CPU test.
Keep backward compatibility tests for legacy configuration and checkpoints.
Use tiny models for integration tests; do not download large weights for unit tests.

```bash
python -m pytest -q
ruff check train.py eval.py sd_aio tests
ruff format --check train.py eval.py sd_aio tests
```

Keep this contract and the concise README aligned with code. Data/model details
belong in `docs/`. Do not change active experiments or overwrite their weights
while refactoring framework code.
