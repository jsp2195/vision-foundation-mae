from pathlib import Path

import torch

from minimae_vit.utils.checkpoint import load_checkpoint, save_checkpoint


def test_ckpt_roundtrip(tmp_path: Path):
    p = tmp_path / "a.pt"
    payload = {"x": torch.tensor([1, 2, 3]), "epoch": 2}
    save_checkpoint(p, payload)
    out = load_checkpoint(p)
    assert out["epoch"] == 2
    assert torch.equal(out["x"], payload["x"])
