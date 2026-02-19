from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms as T

from minimae_vit.eval.linear_probe import ViTClassifier
from minimae_vit.models.vit import ViTEncoder


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    if not cfg.classifier_ckpt or not cfg.image_dir:
        raise ValueError("Set classifier_ckpt and image_dir.")
    payload = torch.load(cfg.classifier_ckpt, map_location="cpu")
    mc = payload["resolved_cfg"]["model"]
    dc = payload["resolved_cfg"]["dataset"]
    encoder = ViTEncoder(
        img_size=mc["img_size"], patch_size=mc["patch_size"], in_chans=mc["in_chans"], embed_dim=mc["embed_dim"],
        depth=mc["depth"], num_heads=mc["num_heads"], mlp_ratio=mc["mlp_ratio"], drop=mc["drop"], attn_drop=mc["attn_drop"]
    )
    model = ViTClassifier(encoder, dim=mc["embed_dim"], num_classes=dc["num_classes"], freeze_encoder=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    tf = T.Compose([T.Resize((dc["img_size"], dc["img_size"])), T.ToTensor()])

    for p in sorted(Path(cfg.image_dir).glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        prob = model(x).softmax(dim=1)
        c = int(prob.argmax(dim=1))
        print(f"{p.name}: top1={c} prob={float(prob[0, c]):.4f}")


if __name__ == "__main__":
    main()
