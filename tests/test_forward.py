import torch
from omegaconf import OmegaConf

from minimae_vit.models.mae import MAE


def test_forward_shapes_and_finite_loss():
    cfg = OmegaConf.create({"model": {"img_size": 32, "patch_size": 8, "in_chans": 3, "embed_dim": 64, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "decoder_dim": 32, "decoder_depth": 1, "decoder_heads": 2, "mask_ratio": 0.75}})
    m = MAE(cfg)
    x = torch.randn(2, 3, 32, 32)
    pred, mask, loss, aux = m(x)
    assert pred.shape[:2] == mask.shape
    assert torch.isfinite(loss)
    for k in ["patch_mean", "patch_var", "ids_restore", "ids_keep"]:
        assert k in aux
