import torch
from torch import nn

from minimae_vit.models.patching import patchify
from minimae_vit.models.vit import Block, ViTEncoder


class MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.patch = m.patch_size
        self.img_size = m.img_size
        self.encoder = ViTEncoder(
            img_size=m.img_size,
            patch_size=m.patch_size,
            in_chans=m.in_chans,
            embed_dim=m.embed_dim,
            depth=m.depth,
            num_heads=m.num_heads,
            mlp_ratio=m.mlp_ratio,
        )
        n = (m.img_size // m.patch_size) ** 2
        self.dec_proj = nn.Linear(m.embed_dim, m.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, m.decoder_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, n + 1, m.decoder_dim))
        self.dec_blocks = nn.ModuleList([Block(m.decoder_dim, m.decoder_heads, m.mlp_ratio) for _ in range(m.decoder_depth)])
        self.dec_norm = nn.LayerNorm(m.decoder_dim)
        self.dec_pred = nn.Linear(m.decoder_dim, m.patch_size * m.patch_size * m.in_chans)
        self.mask_ratio = m.mask_ratio
        self.eps = 1e-6

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        b, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward(self, imgs: torch.Tensor):
        target = patchify(imgs, self.patch)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target_norm = (target - mean) / torch.sqrt(var + self.eps)

        x = self.encoder.patch_embed(imgs).flatten(2).transpose(1, 2)
        x_vis, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        cls = self.encoder.cls_token.expand(imgs.size(0), -1, -1)
        pos = self.encoder.pos_embed[:, 1:, :]
        x_vis = x_vis + torch.gather(pos.expand(imgs.size(0), -1, -1), 1, ids_keep.unsqueeze(-1).expand(-1, -1, pos.size(-1)))
        x_enc = torch.cat([cls + self.encoder.pos_embed[:, :1, :], x_vis], dim=1)
        for blk in self.encoder.blocks:
            x_enc = blk(x_enc)
        x_enc = self.encoder.norm(x_enc)

        x_dec = self.dec_proj(x_enc)
        x_tokens = x_dec[:, 1:, :]
        mask_tokens = self.mask_token.expand(imgs.size(0), ids_restore.shape[1] - x_tokens.shape[1], -1)
        x_ = torch.cat([x_tokens, mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        x_dec = torch.cat([x_dec[:, :1, :], x_], dim=1) + self.dec_pos
        for blk in self.dec_blocks:
            x_dec = blk(x_dec)
        pred = self.dec_pred(self.dec_norm(x_dec[:, 1:, :]))

        loss = ((pred - target_norm) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        aux = {"patch_mean": mean, "patch_var": var, "ids_restore": ids_restore, "ids_keep": ids_keep}
        return pred, mask, loss, aux
