import pytest
import torch

from sd_aio.eval import predict_tiles


class Identity:
    def __init__(self):
        self.shapes = []
        self.ids = []

    def noise_seeds(self, ids):
        self.ids.extend(ids)
        return [1]

    def __call__(self, x, text, noise_seeds):
        self.shapes.append(x.shape[-2:])
        return x


@pytest.mark.parametrize("shape", [(32, 40), (128, 192), (151, 213), (30, 200)])
def test_tiles_cover_edges_and_preserve_identity(shape):
    x = torch.rand(1, 3, *shape) * 2 - 1
    model = Identity()
    out = predict_tiles(model, x, None, "image", 64, 16)
    torch.testing.assert_close(out, x, atol=2e-6, rtol=2e-6)
    assert all(h <= 64 and w <= 64 for h, w in model.shapes)
    if max(shape) <= 64:
        assert model.ids == ["image"]
    else:
        assert len(model.ids) == len(set(model.ids))


@pytest.mark.parametrize("tile,overlap", [(0, 0), (64, -1), (64, 64)])
def test_invalid_tiles(tile, overlap):
    with pytest.raises(ValueError):
        predict_tiles(Identity(), torch.zeros(1, 3, 80, 80), None, "image", tile, overlap)
