from pathlib import Path

import hydra
from omegaconf import OmegaConf

from minimae_vit.eval.finetune import run_finetune
from minimae_vit.utils.io import ensure_dir
from minimae_vit.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="finetune_cifar10")
def main(cfg):
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_root) / cfg.run_name)
    (out_dir / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg))
    acc = run_finetune(cfg, out_dir)
    print(f"finetune_test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
