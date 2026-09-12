"""Inference must reject ambiguous files before writing any predictions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from sd_aio.eval import run_inference


class IdentityRestorer:
    def __init__(self):
        self.calls = []
        self.prompt_calls = 0
        self.prompt_embeddings = {"test": torch.zeros(1, 1, 1)}

    def encode_prompt(self, prompt, **kwargs):
        self.prompt_calls += 1
        return self.prompt_embeddings["test"]

    def noise_seeds(self, ids):
        return [1] * len(ids)

    def __call__(self, tensor, text, noise_seeds):
        self.calls.append(tensor.clone())
        return tensor


def config(preprocessing=None):
    cfg = OmegaConf.create(
        {
            "stage": "spade",
            "mixed_precision": "no",
            "data": {},
            "eval": {"pad_to_multiple": 8, "tiling": False, "patchwise": False},
        }
    )
    if preprocessing is not None:
        cfg.data.preprocessing = {"inference": preprocessing}
    return cfg


def write_image(path, size=(72, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    image.save(path)
    return image


def infer(model, cfg, input_path, output, gt=None):
    stage = SimpleNamespace(set_eval_mode=lambda model: None)
    return run_inference(
        stage,
        model,
        model,
        cfg,
        input_path,
        output,
        device=torch.device("cpu"),
        prompt="clean image",
        gt_dir=gt,
    )


@pytest.mark.parametrize("case", ["duplicate_gt", "duplicate_output", "missing_gt"])
def test_ambiguities_fail_before_prompt_forward_or_output_creation(tmp_path, case):
    lq, gt, output = tmp_path / "lq", tmp_path / "gt", tmp_path / "output"
    write_image(lq / "a.png")
    write_image(gt / "a.png")
    if case == "duplicate_gt":
        write_image(gt / "nested" / "a.jpg")
    elif case == "duplicate_output":
        write_image(lq / "a.jpg")
    else:
        write_image(lq / "b.png")
    model = IdentityRestorer()
    with pytest.raises((ValueError, FileNotFoundError)):
        infer(model, config(), lq, output, gt)
    assert model.calls == []
    assert model.prompt_calls == 0
    assert not output.exists()


def test_same_stem_in_different_input_folders_has_distinct_outputs(tmp_path):
    lq = tmp_path / "lq"
    write_image(lq / "scene1" / "a.png")
    write_image(lq / "scene2" / "a.png")
    model = IdentityRestorer()
    files, report = infer(model, config(), lq, tmp_path / "output")
    assert len(files) == len(set(files)) == 2
    assert [path.parent.name for path in files] == ["scene1", "scene2"]
    assert report is None


def test_default_inference_preserves_native_pixels_and_dimensions(tmp_path):
    path = tmp_path / "lq.png"
    image = write_image(path, size=(74, 51))
    model = IdentityRestorer()
    files, _ = infer(model, config(), path, tmp_path / "output")
    expected = torch.as_tensor(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
    assert torch.equal(model.calls[0][0, :, :51, :74], expected)
    with Image.open(files[0]) as result:
        assert result.size == image.size


@pytest.mark.parametrize("mode", ["center_crop", "resize_short_center_crop", "resize"])
def test_explicit_inference_geometry_is_shared_with_gt(tmp_path, mode):
    lq, gt = tmp_path / "lq", tmp_path / "gt"
    image = write_image(lq / "a.png")
    gt.mkdir()
    image.save(gt / "a.png")
    model = IdentityRestorer()
    cfg = config({"mode": mode, "image_size": 32, "interpolation": "bicubic"})
    files, report = infer(model, cfg, lq, tmp_path / "output", gt)
    assert model.calls[0].shape == (1, 3, 32, 32)
    assert report.overall["psnr"] >= 90
    assert report.overall["ssim"] == pytest.approx(1)
    with Image.open(files[0]) as result:
        assert result.size == (32, 32)


def test_invalid_inference_geometry_fails_before_forward(tmp_path):
    image = tmp_path / "a.png"
    write_image(image)
    model = IdentityRestorer()
    with pytest.raises(ValueError, match="random_crop"):
        infer(model, config({"mode": "random_crop", "image_size": 32}), image, tmp_path / "output")
    assert not model.calls
    assert not (tmp_path / "output").exists()
