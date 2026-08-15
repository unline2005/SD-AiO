from omegaconf import OmegaConf

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


def test_run_inference_pads_crops_and_saves(tmp_path):
    task = make_synthetic_task(tmp_path, "Test_Denoise_15", "noise", n_images=2)
    cfg = _cfg(tmp_path, task)
    model = make_tiny_restorer("simple")

    import sd_aio.spade as stage

    input_image = tmp_path / "Test_Denoise_15_lq" / "0000.png"
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
