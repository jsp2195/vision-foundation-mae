from __future__ import annotations

import hydra
from omegaconf import DictConfig

from minimae_vit.data.datamodules import build_dataloaders
from minimae_vit.models.mae import MAE
from minimae_vit.train.loops import pretrain_mae
from minimae_vit.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="mae_cifar10")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    dataloaders = build_dataloaders(cfg)
    model = MAE(cfg.model)
    out = pretrain_mae(cfg, model, dataloaders)
    print(f"MAE training complete. Outputs: {out}")


if __name__ == "__main__":
    main()
