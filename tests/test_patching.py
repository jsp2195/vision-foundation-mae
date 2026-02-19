import torch

from minimae_vit.models.patching import patchify, unpatchify


def test_patch_roundtrip():
    x = torch.randn(2, 3, 32, 32)
    p = patchify(x, 8)
    xr = unpatchify(p, 8, 32)
    assert torch.allclose(x, xr)
