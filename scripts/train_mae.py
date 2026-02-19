from pathlib import Path

import hydra
from omegaconf import OmegaConf

from minimae_vit.train.loops import pretrain_mae
from minimae_vit.utils.env import env_summary
from minimae_vit.utils.io import ensure_dir
from minimae_vit.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="mae_cifar10")
def main(cfg):
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_root) / cfg.run_name)
    (out_dir / "manifest.yaml").write_text(OmegaConf.to_yaml(cfg) + "\n# env\n" + str(env_summary()))
    pretrain_mae(cfg, out_dir)


if __name__ == "__main__":
    main()
