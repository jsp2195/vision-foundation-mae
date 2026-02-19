from pathlib import Path

import torch

from minimae_vit.utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    p = tmp_path / "ckpt.pt"
    payload = {"a": torch.tensor([1.0]), "epoch": 1}
    save_checkpoint(p, payload)
    out = load_checkpoint(p)
    assert out["epoch"] == 1
    assert torch.equal(out["a"], payload["a"])
