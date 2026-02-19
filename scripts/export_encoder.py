from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from minimae_vit.utils.io import ensure_dir


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    if not cfg.get("ckpt_path"):
        raise ValueError("set ckpt_path=/path/to/checkpoint.pt")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt.get("encoder_state")
    if state is None and "model_state" in ckpt:
        state = {k.replace("encoder.", "", 1): v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    out_dir = ensure_dir(Path(cfg.output_root) / cfg.run_name)
    torch.save({"resolved_cfg": OmegaConf.to_container(cfg, resolve=True), "state_dict": state}, out_dir / "vit_encoder_backbone.pt")


if __name__ == "__main__":
    main()
