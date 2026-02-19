import torch
from omegaconf import OmegaConf

from minimae_vit.models.mae import MAE


def _cfg():
    return OmegaConf.create({
        "img_size": 64, "patch_size": 8, "in_chans": 3, "embed_dim": 64, "depth": 2, "num_heads": 4,
        "mlp_ratio": 2.0, "drop": 0.0, "attn_drop": 0.0, "mask_ratio": 0.75,
        "dec_embed_dim": 32, "dec_depth": 1, "dec_heads": 4, "norm_eps": 1e-6,
    })


def test_random_masking_shapes():
    model = MAE(_cfg())
    x = torch.randn(4, model.num_patches, 64)
    vis, mask, ids_restore, ids_keep = model.random_masking(x, 0.75)
    assert vis.shape[1] == int(model.num_patches * 0.25)
    assert mask.shape == (4, model.num_patches)
    assert ids_restore.shape == (4, model.num_patches)
    assert ids_keep.shape[1] == vis.shape[1]
