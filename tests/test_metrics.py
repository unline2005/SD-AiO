import torch

from sd_aio import metrics


def test_to_numpy_rgb_range_and_shape():
    image = torch.tensor([[[-1.0]], [[0.0]], [[1.0]]])
    array = metrics.to_numpy_rgb(image)
    assert array.shape == (1, 1, 3)
    assert array[0, 0, 0] == 0.0
    assert array[0, 0, 1] == 0.5
    assert array[0, 0, 2] == 1.0


def test_identical_images_have_perfect_psnr_ssim():
    image = torch.rand(1, 3, 32, 32) * 2 - 1
    nearly_identical = image.clone()
    nearly_identical[0, 0, 0, 0] += 1e-6
    assert metrics.compute_psnr(nearly_identical, image) > 40
    assert metrics.compute_ssim(nearly_identical, image) > 0.999


def test_task_equal_overall_is_unweighted():
    accumulator = metrics.MetricAccumulator(metric_names=("psnr",))
    for _ in range(10):
        accumulator.add("a", {"psnr": 10.0})
    accumulator.add("b", {"psnr": 20.0})
    assert accumulator.per_task() == {"a": {"psnr": 10.0}, "b": {"psnr": 20.0}}
    assert accumulator.overall() == {"psnr": 15.0}
