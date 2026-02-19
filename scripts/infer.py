from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from minimae_vit.models.vit import ViTEncoder


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    if not cfg.get("ckpt_path") or not cfg.get("image_dir"):
        raise ValueError("set ckpt_path and image_dir")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    enc = ViTEncoder(**OmegaConf.to_container(cfg.model, resolve=True))
    enc.load_state_dict(ckpt["encoder_state"], strict=False)
    head = torch.nn.Linear(cfg.model.embed_dim, cfg.dataset.num_classes)
    head.load_state_dict(ckpt["head_state"])
    enc.eval(); head.eval()
    tf = transforms.Compose([transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)), transforms.ToTensor()])
    for p in sorted(Path(cfg.image_dir).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        probs = head(enc(x)[:, 0]).softmax(dim=-1)[0]
        pred = int(probs.argmax().item())
        print(f"{p.name}: class={pred} prob={probs[pred].item():.4f}")


if __name__ == "__main__":
    main()
