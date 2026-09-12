import pytest
from omegaconf import OmegaConf
from PIL import Image

from sd_aio import data
from sd_aio import eval as sd_eval
from tests.helpers import make_synthetic_task, make_tiny_restorer


def _cfg(tmp_path, task):
    return OmegaConf.create(
        {
            "stage": "spade",
            "mixed_precision": "no",
            "persistent_workers": False,
            "pin_memory": False,
            "output_dir": str(tmp_path / "out"),
            "data": {
                "num_workers": 0,
                "paired_sampling": "uniform",
                "train_image_size": 64,
                "augmentation": {
                    "hflip_prob": 0.0,
                    "vflip_prob": 0.0,
                    "rot90_prob": 0.0,
                },
                "train": [task],
                "test": [task],
            },
            "trainer": {
                "train_batch_size": 1,
                "num_images_save_eval": 2,
                "log_every": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "max_grad_norm": 1.0,
                "eval_freq": 0,
                "checkpointing_steps": 0,
                "round_robin": False,
            },
            "eval": {
                "batch_size": 1,
                "crop_to_multiple": 16,
                "pad_to_multiple": 64,
                "num_samples_per_task": None,
                "tiling": False,
                "patchwise": False,
                "tile_overlap": 0,
                "tile_size": 32,
                "compute_lpips": False,
            },
        }
    )


def test_run_eval_uses_crop_pad_protocol_and_task_equal_overall(tmp_path):
    task = make_synthetic_task(tmp_path, "Test_Denoise_15", "noise", n_images=3)
    cfg = _cfg(tmp_path, task)
    _, test_loaders = data.build_loaders(cfg, verbose=False)
    model = make_tiny_restorer("simple")
    model.prompt_embeddings[task["name"]] = model.prompt_embeddings["Task"].clone()

    import sd_aio.spade as stage

    report = sd_eval.run_eval(
        stage,
        model,
        model,
        test_loaders,
        cfg,
        device=model.prompt_embeddings["Task"].device,
        weight_dtype=model.prompt_embeddings["Task"].dtype,
        step=7,
        output_dir=cfg.output_dir,
        num_samples_per_task=1,
    )
    assert report.step == 7
    assert "Test_Denoise_15" in report.task_metrics
    assert "psnr" in report.task_metrics["Test_Denoise_15"]
    assert "psnr" in report.overall
    assert len(report.vis_paths) == 1


@pytest.mark.parametrize("patchwise", [False, True])
def test_run_inference_pads_crops_and_saves(tmp_path, patchwise):
    task = make_synthetic_task(tmp_path, "Test_Denoise_15", "noise", n_images=2)
    cfg = _cfg(tmp_path, task)
    model = make_tiny_restorer("simple")

    import sd_aio.spade as stage

    input_image = tmp_path / "Test_Denoise_15_lq" / "0000.png"
    cfg.eval.patchwise = patchwise
    cfg.eval.tile_size = 64
    save_dir = tmp_path / "inference"
    paths, report = sd_eval.run_inference(
        stage,
        model,
        model,
        cfg,
        input_image,
        save_dir,
        device=model.prompt_embeddings["Task"].device,
        weight_dtype=model.prompt_embeddings["Task"].dtype,
    )
    assert len(paths) == 1
    assert paths[0].exists()
    with Image.open(paths[0]) as result, Image.open(input_image) as original:
        assert result.size == original.size
    assert report is None


def test_run_inference_with_gt_scores(tmp_path):
    task = make_synthetic_task(tmp_path, "Test_Haze", "haze", n_images=2)
    cfg = _cfg(tmp_path, task)
    model = make_tiny_restorer("simple")

    import sd_aio.spade as stage

    paths, report = sd_eval.run_inference(
        stage,
        model,
        model,
        cfg,
        task["lq_path"],
        tmp_path / "inference",
        device=model.prompt_embeddings["Task"].device,
        weight_dtype=model.prompt_embeddings["Task"].dtype,
        gt_dir=task["gt_path"],
    )
    assert len(paths) == 2
    assert report is not None
    assert "psnr" in report.overall


def test_classifier_eval_limits_remaining_batch_and_saves_named_metrics(tmp_path):
    import json
    from collections import OrderedDict
    from types import SimpleNamespace

    import torch

    from sd_aio import classifier

    class RawClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, images):
            self.batch_sizes.append(images.shape[0])
            return images

    class PreparedModel(torch.nn.Module):
        def forward(self, images):
            raise AssertionError("Rank-zero evaluation must not call the DDP wrapper")

    raw_model = RawClassifier()
    names = ["haze", "rain", "snow", "lowlight"]
    labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.long)
    batch = {"lq": labels.clone().float(), "label": labels}
    stage = SimpleNamespace(
        set_eval_mode=lambda model: model.eval(),
        eval_step=lambda model, raw, item: {
            "predictions": model(item["lq"]).long(),
            "labels": item["label"],
        },
        compute_binary_metrics=classifier.compute_binary_metrics,
    )
    cfg = OmegaConf.create(
        {"stage": "classifier", "mixed_precision": "no", "data": {"deg_types": names}, "eval": {}}
    )
    report = sd_eval.run_eval(
        stage,
        PreparedModel(),
        raw_model,
        OrderedDict([("first", [batch, batch]), ("second", [batch, batch])]),
        cfg,
        device=torch.device("cpu"),
        num_samples_per_task=5,
        step=3,
    )
    assert raw_model.batch_sizes == [4, 1, 4, 1]
    assert report.classification["num_samples"] == 10
    assert report.classification["class_names"] == names
    assert [entry["name"] for entry in report.classification["per_class"]] == names
    assert report.classification["exact_match"] == 1.0
    path = sd_eval.save_report(report, tmp_path / "classification.json")
    saved = json.loads(path.read_text())
    assert saved["classification"] == report.classification
    assert saved["classification"]["per_class"][1]["f1"] == 1.0


def test_classifier_eval_rejects_class_count_mismatch():
    from collections import OrderedDict
    from types import SimpleNamespace

    import pytest
    import torch

    from sd_aio import classifier

    model = torch.nn.Identity()
    labels = torch.zeros(2, 4, dtype=torch.long)
    stage = SimpleNamespace(
        set_eval_mode=lambda raw: raw.eval(),
        eval_step=lambda prepared, raw, batch: {"predictions": labels, "labels": labels},
        compute_binary_metrics=classifier.compute_binary_metrics,
    )
    cfg = OmegaConf.create(
        {"stage": "classifier", "mixed_precision": "no", "data": {"deg_types": ["rain"]}, "eval": {}}
    )
    with pytest.raises(ValueError, match=r"data\.deg_types"):
        sd_eval.run_eval(
            stage,
            model,
            model,
            OrderedDict([("test", [{"lq": torch.zeros(2, 3, 16, 16), "label": labels}])]),
            cfg,
            device=torch.device("cpu"),
        )
