from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from minimae_vit.models.mae import MAE
from minimae_vit.utils.checkpoint import load_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    if not cfg.pretrained_ckpt:
        raise ValueError("Set pretrained_ckpt=<path-to-mae-checkpoint>.")
    ckpt = load_checkpoint(cfg.pretrained_ckpt)
    mae = MAE(cfg.model)
    mae.load_state_dict(ckpt["model_state"], strict=False)
    out = Path(cfg.output_root) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "encoder_state_dict": mae.encoder.state_dict(),
    }
    path = out / "vit_encoder_backbone.pt"
    torch.save(payload, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
