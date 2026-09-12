# Data and preprocessing

All model-facing images are RGB tensors in `[-1, 1]`. Paired batches contain
`lq`, `gt`, `task_name`, `image_id` and `deg_type`. Classifier batches contain
`lq` and multi-hot `label`; label order comes from `data.deg_types`.

## Directory tasks

```yaml
name: train_haze
deg_type: haze
lq_path: /data/haze/lq
gt_path: /data/haze/gt
sampling_weight: 1.0
prompt: a high quality clean image
```

Default pairing uses equal filename stems. `match_prefix_len` supports the legacy
GT prefix convention. `pairing: relative_stem` with `recursive: true` supports
nested directories. Ambiguous keys, missing images, empty tasks and mismatched
paired dimensions raise errors instead of skipping data.

## JSON and JSONL manifests

```yaml
name: train_haze
deg_type: haze
manifest: /data/manifests/haze.jsonl
manifest_root: ../images
```

Each JSONL line contains one pair:

```json
{"lq":"haze/0001.png","gt":"clean/0001.png"}
```

JSON uses a list of the same records. Paths resolve relative to the manifest
directory, or to `manifest_root` when specified. Classification manifests can
provide only `lq`; labels are task-level, not per-record. CSV, LMDB, WebDataset and
remote streaming datasets are not implemented.

## Geometry

Select defaults separately for paired/classification training, evaluation and
direct inference. Task-local `preprocessing` overrides dataset-loader defaults.

```yaml
data:
  preprocessing:
    paired_train: {mode: random_crop, interpolation: bicubic, exif_transpose: false}
    paired_eval: {mode: native, interpolation: bicubic, exif_transpose: false}
    classifier_train: {mode: random_crop, interpolation: bilinear, exif_transpose: true}
    classifier_eval: {mode: center_crop, interpolation: bilinear, exif_transpose: true}
    inference: {mode: native, image_size: 512, interpolation: bicubic, exif_transpose: false}
```

| Mode | Geometry |
|---|---|
| `native` | Preserve original size |
| `random_crop` | Enlarge undersized images if needed, then random square crop |
| `center_crop` | Enlarge undersized images if needed, then centered square crop |
| `resize_short_center_crop` | Resize the short side to the target, then centered square crop |
| `resize` | Resize to the configured square size |

Use `image_size` to override the target within a preprocessing block; otherwise
dataset loaders use `data.train_image_size` or `data.image_size`. LQ and GT share
geometry and augmentation decisions. Native images of different sizes require
batch size 1. Benchmark evaluation subsequently applies the explicit
`eval.crop_to_multiple` / `eval.pad_to_multiple` protocol.

Direct `eval.py --input` uses `data.preprocessing.inference`; it does not silently
inherit a training crop. Duplicate output stems and ambiguous/missing GT stems
are checked before inference writes files.

## Sampling and online noise

Paired and classifier loaders support uniform and task-balanced sampling.
`sampling_weight` is a task's total probability mass, independent of image count.
To balance sources, give each task weight `1 / tasks_in_its_source`.

Online Gaussian noise uses `sigma / 127.5` in the shared range. Evaluation uses a
deterministic seed. This is clipped float noise, not an implicit reproduction of
every benchmark's quantized noise-generation protocol.

Do not infer leakage freedom from directory names. Keep scene/GT identities
disjoint across splits and save the selected manifests with the experiment.
