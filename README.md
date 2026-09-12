# SD-AiO

All-in-one image restoration research code built with PyTorch, Transformers,
Diffusers and Accelerate. **Work in progress: the final method is not yet implemented.**

## Structure

- `train.py`: one training loop for all stages.
- `eval.py`, `sd_aio/eval.py`: benchmark and inference entry point / implementation.
- `sd_aio/data.py`: datasets, manifests, preprocessing and sampling.
- `sd_aio/classifier.py`, `vae_encoder.py`, `spade.py`: current stage implementations.
- `sd_aio/backends.py`: model-family contracts and latent/prediction helpers.
- `sd_aio/config.py`, `checkpoint.py`: experiment composition and persistence.
- `sd_aio/optim.py`, `losses.py`, `runtime.py`: optimizers, pixel losses and logging.

## Check the pipeline

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check train.py eval.py sd_aio tests
python tools/make_smoke_project.py /tmp/sdaio-smoke
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python train.py --config /tmp/sdaio-smoke/classifier.yaml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python train.py --config /tmp/sdaio-smoke/vae.yaml
```

The smoke example uses tiny random models and requires no weight downloads.
Use a new directory when repeating it. Experiment settings live in YAML;
`key=value` CLI overrides and `_base_` inheritance are supported.

The current restoration backend is SD UNet. SD3/FLUX integration is deferred;
only their tested mathematical/layout helpers are present, not complete pipelines.

References: [data](docs/data.md), [experiments](docs/experiments.md),
[extension interfaces](docs/extending.md), [quality criteria](docs/quality.md).
