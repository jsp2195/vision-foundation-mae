import torch


def patchify(imgs: torch.Tensor, patch: int) -> torch.Tensor:
    b, c, h, w = imgs.shape
    assert h == w and h % patch == 0
    n = h // patch
    x = imgs.reshape(b, c, n, patch, n, patch)
    x = torch.einsum("bcnphq->bnhpqc", x)
    return x.reshape(b, n * n, patch * patch * c)


def unpatchify(patches: torch.Tensor, patch: int, img_size: int, chans: int = 3) -> torch.Tensor:
    b, n, d = patches.shape
    h = w = img_size // patch
    assert h * w == n
    x = patches.reshape(b, h, w, patch, patch, chans)
    x = torch.einsum("bnhpqc->bcnphq", x)
    return x.reshape(b, chans, img_size, img_size)
