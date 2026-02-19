from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from minimae_vit.data.datamodules import build_dataloaders
from minimae_vit.models.mae import MAE
from minimae_vit.models.vit import ViTEncoder
from minimae_vit.train.losses import cross_entropy_loss
from minimae_vit.utils.checkpoint import save_checkpoint
from minimae_vit.utils.logging import RunLogger
from minimae_vit.utils.metrics import accuracy


def _device(cfg):
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _amp_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(name, torch.float16)


def pretrain_mae(cfg, out_dir: Path):
    device = _device(cfg)
    loaders = build_dataloaders(cfg)
    model = MAE(cfg).to(device)
    if cfg.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    logger = RunLogger(out_dir)
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        bar = tqdm(loaders["pretrain"], desc=f"mae {epoch}")
        for x, _ in bar:
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg.amp_dtype), enabled=device.type == "cuda"):
                _, _, loss, _ = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % cfg.train.log_every == 0:
                logger.log(step, {"train/loss": float(loss.item()), "train/lr": opt.param_groups[0]["lr"]})
                bar.set_postfix(loss=float(loss.item()))
        save_checkpoint(
            out_dir / "checkpoints" / "last.pt",
            {
                "resolved_cfg": OmegaConf.to_container(cfg, resolve=True),
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "global_step": step,
            },
        )
    logger.close()


def run_classifier(cfg, out_dir: Path, finetune: bool = False):
    device = _device(cfg)
    loaders = build_dataloaders(cfg)
    enc = ViTEncoder(**OmegaConf.to_container(cfg.model, resolve=True, enum_to_str=True))
    if cfg.train.encoder_ckpt:
        ckpt = torch.load(cfg.train.encoder_ckpt, map_location="cpu")
        state = ckpt.get("encoder_state", ckpt.get("model_state", ckpt))
        missing, _ = enc.load_state_dict(state, strict=False)
        if missing:
            pass
    enc.to(device)
    if not finetune:
        for p in enc.parameters():
            p.requires_grad = False
    head = torch.nn.Linear(cfg.model.embed_dim, cfg.dataset.num_classes).to(device)
    params = list(head.parameters()) + (list(enc.parameters()) if finetune else [])
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = RunLogger(out_dir)
    best = 0.0
    for epoch in range(cfg.train.epochs):
        enc.train(finetune)
        head.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            feats = enc(x)[:, 0]
            logits = head(feats)
            loss = cross_entropy_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        enc.eval()
        head.eval()
        accs = []
        with torch.no_grad():
            for x, y in loaders["test"]:
                x, y = x.to(device), y.to(device)
                logits = head(enc(x)[:, 0])
                accs.append(accuracy(logits, y))
        test_acc = float(sum(accs) / len(accs))
        logger.log(epoch, {"eval/acc": test_acc})
        if test_acc >= best:
            best = test_acc
            save_checkpoint(out_dir / "checkpoints" / "best.pt", {"encoder_state": enc.state_dict(), "head_state": head.state_dict(), "acc": best, "resolved_cfg": OmegaConf.to_container(cfg, resolve=True)})
    save_checkpoint(out_dir / "checkpoints" / "last.pt", {"encoder_state": enc.state_dict(), "head_state": head.state_dict(), "acc": best, "resolved_cfg": OmegaConf.to_container(cfg, resolve=True)})
    logger.close()
    return best
