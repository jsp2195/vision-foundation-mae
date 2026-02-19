from __future__ import annotations

import hydra
from omegaconf import DictConfig

from minimae_vit.data.datamodules import build_dataloaders
from minimae_vit.models.mae import MAE
from minimae_vit.train.loops import train_classifier
from minimae_vit.utils.checkpoint import load_checkpoint
from minimae_vit.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="finetune_cifar10")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    if not cfg.pretrained_ckpt:
        raise ValueError("Set pretrained_ckpt=<path-to-mae-checkpoint>.")
    ckpt = load_checkpoint(cfg.pretrained_ckpt)
    mae = MAE(cfg.model)
    mae.load_state_dict(ckpt["model_state"], strict=False)
    dataloaders = build_dataloaders(cfg)
    out, best = train_classifier(cfg, mae.encoder, dataloaders, freeze_encoder=False)
    print(f"Finetune done. best_acc={best:.4f} outputs={out}")


if __name__ == "__main__":
    main()
