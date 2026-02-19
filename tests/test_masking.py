import torch
from omegaconf import OmegaConf

from minimae_vit.models.mae import MAE


def test_masking_shapes_and_ratio():
    cfg = OmegaConf.create({"model": {"img_size": 32, "patch_size": 8, "in_chans": 3, "embed_dim": 64, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "decoder_dim": 32, "decoder_depth": 1, "decoder_heads": 2, "mask_ratio": 0.75}})
    m = MAE(cfg)
    t = torch.randn(4, 16, 64)
    xv, mask, ids_restore, ids_keep = m.random_masking(t, 0.75)
    assert xv.shape[1] == 4
    assert mask.shape == (4, 16)
    assert ids_restore.shape == (4, 16)
    assert ids_keep.shape == (4, 4)
