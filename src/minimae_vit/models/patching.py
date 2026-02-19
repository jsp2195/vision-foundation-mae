from __future__ import annotations

import torch


def patchify(imgs: torch.Tensor, patch: int) -> torch.Tensor:
    b, c, h, w = imgs.shape
    if h % patch != 0 or w % patch != 0:
        raise ValueError(f"Image size {(h, w)} must be divisible by patch {patch}.")
    gh, gw = h // patch, w // patch
    x = imgs.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, gh * gw, patch * patch * c)


def unpatchify(patches: torch.Tensor, patch: int, img_size: int, chans: int = 3) -> torch.Tensor:
    b, n, _ = patches.shape
    gh = gw = img_size // patch
    if n != gh * gw:
        raise ValueError(f"Patch count {n} incompatible with img_size={img_size}, patch={patch}.")
    x = patches.reshape(b, gh, gw, patch, patch, chans)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.reshape(b, chans, img_size, img_size)
