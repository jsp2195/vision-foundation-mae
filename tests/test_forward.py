import torch
from omegaconf import OmegaConf

from minimae_vit.models.mae import MAE


def test_forward_shapes_and_finite_loss():
    cfg = OmegaConf.create({
        "img_size": 64, "patch_size": 8, "in_chans": 3, "embed_dim": 64, "depth": 2, "num_heads": 4,
        "mlp_ratio": 2.0, "drop": 0.0, "attn_drop": 0.0, "mask_ratio": 0.75,
        "dec_embed_dim": 32, "dec_depth": 1, "dec_heads": 4, "norm_eps": 1e-6,
    })
    m = MAE(cfg)
    loss, pred, mask, aux = m(torch.randn(2, 3, 64, 64))
    assert torch.isfinite(loss)
    assert pred.shape[:2] == mask.shape
    for k in ["patch_mean", "patch_var", "ids_restore", "ids_keep"]:
        assert k in aux
