import torch

from minimae_vit.models.patching import patchify, unpatchify


def test_patch_round_trip():
    x = torch.rand(2, 3, 64, 64)
    p = patchify(x, patch=8)
    xr = unpatchify(p, patch=8, img_size=64)
    assert torch.allclose(x, xr)
