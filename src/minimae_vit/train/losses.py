import torch


def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, y)
