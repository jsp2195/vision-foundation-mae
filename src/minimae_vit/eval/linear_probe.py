from __future__ import annotations

from torch import nn


class ViTClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, dim: int, num_classes: int = 10, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, num_classes)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z[:, 0])
