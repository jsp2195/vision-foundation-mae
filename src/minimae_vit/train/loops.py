from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from minimae_vit.eval.linear_probe import ViTClassifier
from minimae_vit.train.schedulers import cosine_lr
from minimae_vit.utils.checkpoint import save_checkpoint
from minimae_vit.utils.env import env_summary
from minimae_vit.utils.logging import RunLogger


def _device(cfg) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _run_dir(cfg) -> Path:
    d = Path(cfg.output_root) / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def pretrain_mae(cfg, model, dataloaders):
    device = _device(cfg)
    model.to(device)
    if cfg.compile:
        model = torch.compile(model)
    out = _run_dir(cfg)
    (out / "checkpoints").mkdir(exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    (out / "manifest.json").write_text(OmegaConf.to_yaml(OmegaConf.create(env_summary())))
    logger = RunLogger(out, cfg.logging.tensorboard)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    total = len(dataloaders["pretrain"]) * cfg.train.epochs
    warm = len(dataloaders["pretrain"]) * cfg.train.warmup_epochs

    step = 0
    best = float("inf")
    for epoch in range(cfg.train.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for it, (x, _) in enumerate(tqdm(dataloaders["pretrain"], desc=f"pretrain {epoch}")):
            x = x.to(device)
            lr = cosine_lr(step, total, cfg.train.lr, warm)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss, _, _, _ = model(x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if it % cfg.train.log_interval == 0:
                logger.log(step, {"loss": float(loss.item()), "lr": float(lr)})
            step += 1
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                break
        epoch_loss = sum(losses) / max(1, len(losses))
        logger.log(step, {"epoch": epoch, "epoch_loss": epoch_loss, "epoch_time": time.time() - t0})
        payload = {
            "resolved_cfg": OmegaConf.to_container(cfg, resolve=True),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "global_step": step,
        }
        save_checkpoint(out / "checkpoints" / "last.pt", payload)
        if epoch_loss < best:
            best = epoch_loss
            save_checkpoint(out / "checkpoints" / "best.pt", payload)
        if epoch % cfg.train.save_every == 0:
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch:03d}.pt", payload)
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break
    logger.close()
    return out


def train_classifier(cfg, encoder, dataloaders, freeze_encoder: bool) -> tuple[Path, float]:
    device = _device(cfg)
    out = _run_dir(cfg)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    dim = encoder.pos_embed.shape[-1]
    model = ViTClassifier(encoder, dim, cfg.dataset.num_classes, freeze_encoder=freeze_encoder).to(device)
    params = model.head.parameters() if freeze_encoder else model.parameters()
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    logger = RunLogger(out, cfg.logging.tensorboard)

    best = 0.0
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        for x, y in dataloaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            logger.log(step, {"loss": float(loss.item())})
            step += 1

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dataloaders["test"]:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        logger.log(step, {"epoch": epoch, "accuracy": acc})
        payload = {
            "resolved_cfg": OmegaConf.to_container(cfg, resolve=True),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "accuracy": acc,
        }
        save_checkpoint(out / "checkpoints" / "last.pt", payload)
        if acc >= best:
            best = acc
            save_checkpoint(out / "checkpoints" / "best.pt", payload)
    logger.close()
    return out, best
