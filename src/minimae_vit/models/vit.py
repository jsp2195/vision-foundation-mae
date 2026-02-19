from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        n = (img_size // patch_size) ** 2
        self.num_patches = n
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
