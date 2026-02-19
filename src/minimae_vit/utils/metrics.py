from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)
