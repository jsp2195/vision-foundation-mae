from __future__ import annotations

import torch
from torch import nn

from minimae_vit.models.patching import patchify
from minimae_vit.models.vit import Block, ViTEncoder


class MAE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.drop,
            attn_drop=cfg.attn_drop,
        )
        self.num_patches = self.encoder.num_patches
        self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_chans
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, cfg.dec_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                Block(cfg.dec_embed_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.drop, cfg.attn_drop)
                for _ in range(cfg.dec_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(cfg.dec_embed_dim)
        self.decoder_pred = nn.Linear(cfg.dec_embed_dim, self.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones((b, n), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_keep, mask, ids_restore, ids_keep

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        target = patchify(imgs, patch=self.cfg.patch_size)
        patch_mean = target.mean(dim=-1, keepdim=True)
        patch_var = target.var(dim=-1, keepdim=True)
        target_norm = (target - patch_mean) / torch.sqrt(patch_var + self.cfg.norm_eps)

        x = self.encoder.patch_embed(imgs).flatten(2).transpose(1, 2)
        x_vis, mask, ids_restore, ids_keep = self.random_masking(x, self.cfg.mask_ratio)

        b = imgs.shape[0]
        pos = self.encoder.pos_embed[:, 1:, :].expand(b, -1, -1)
        pos_vis = torch.gather(pos, 1, ids_keep.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
        cls = self.encoder.cls_token.expand(b, -1, -1) + self.encoder.pos_embed[:, :1, :]
        x = torch.cat([cls, x_vis + pos_vis], dim=1)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        x = self.decoder_embed(x)
        cls_token, tokens = x[:, :1], x[:, 1:]
        mask_tokens = self.mask_token.expand(b, self.num_patches - tokens.shape[1], -1)
        filled = torch.cat([tokens, mask_tokens], dim=1)
        filled = torch.gather(filled, 1, ids_restore.unsqueeze(-1).expand(-1, -1, filled.shape[-1]))
        x = torch.cat([cls_token + self.decoder_pos_embed[:, :1], filled + self.decoder_pos_embed[:, 1:]], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x[:, 1:])

        loss = ((pred - target_norm) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        aux = {
            "patch_mean": patch_mean,
            "patch_var": patch_var,
            "ids_restore": ids_restore,
            "ids_keep": ids_keep,
        }
        return loss, pred, mask, aux
