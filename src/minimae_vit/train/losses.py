from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = ((pred - target) ** 2).mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
