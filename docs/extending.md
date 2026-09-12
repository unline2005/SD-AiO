# Adding stages and model backends

Prefer a small `nn.Module` composed from existing modules. Add a class or registry
only when multiple real implementations need it; do not copy the training loop.

## Stage interface

A stage module exports six functions:

```python
build_model(cfg, device=None)
make_optimizer(model, cfg)
set_train_mode(model)
set_eval_mode(model)
compute_loss(model, raw_model, batch, cfg)
eval_step(model, raw_model, batch)
```

Use the prepared `model` for trainable forward/backward and `raw_model` for
attributes and saving. Preserve explicit train/eval modes for frozen components.
Optional `validate_resume` and `before_optimizer_step` hooks are used only where
the stage needs them. Model auxiliary-save hooks persist frozen condition state.

## Actual backend support

| Family | Current status |
|---|---|
| SD2.1 UNet | Existing trained restoration path, checkpoint compatibility and CPU integration tests |
| SD-Turbo | Component metadata fits `sd_unet`; full pretrained restoration quality not validated |
| SD3 | Tested flow/latent primitives; complete restoration adapter not implemented |
| FLUX | Tested flow/latent primitives and 2×2 packing; complete restoration adapter not implemented |

`model.backend: sd_unet` accepts supported UNet metadata and epsilon, velocity or
sample prediction. Unsupported architecture/conditioning is rejected before
loading large weights. Flow prediction is not diffusion `v_prediction`.

`backends.py` provides VAE scale/shift transforms, diffusion-to-clean and
flow-to-clean formulas, and FLUX packing/unpacking. These are reusable numerical
components, not a fabricated end-to-end pipeline.

For a new backend, implement and test all of these before marking it supported:

1. Encoder/decoder latent channels, scaling, shifting and spatial packing.
2. Scheduler time/sigma convention, model prediction target and reconstruction.
3. Text/pooled embeddings, masks, image/text position IDs and guidance inputs.
4. Condition injection locations and dimensions for the actual denoiser blocks.
5. Trainable parameter selection, frozen-module modes and LoRA attachment.
6. Forward/backward, single/DDP equivalence, save/load and resume on tiny fixtures.
7. A real-weight inference test with documented quality, precision and memory.

The current SD-Turbo compatibility does not reproduce its official Euler/ADD
pipeline. Its one-step time and restoration training protocol need separate
validation. SD3 requires multi-encoder text conditioning; FLUX additionally
requires packed spatial tokens and position/guidance handling.

References: [Diffusers DDPM](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py),
[SD3 pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py),
[FLUX pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py),
[SD-Turbo scheduler](https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json).
