import torch
from torch.utils.data import DataLoader

from minimae_vit.data.datasets import build_pretrain_dataset, build_train_test_datasets
from minimae_vit.data.transforms import build_eval_transform, build_train_transform


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    train_tf = build_train_transform(cfg.dataset.img_size)
    eval_tf = build_eval_transform(cfg.dataset.img_size)

    pretrain_ds = build_pretrain_dataset(cfg, train_tf)
    train_ds, test_ds = build_train_test_datasets(cfg, train_tf, eval_tf)

    common = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    return {
        "pretrain": DataLoader(pretrain_ds, shuffle=True, **common),
        "train": DataLoader(train_ds, shuffle=True, **common),
        "test": DataLoader(test_ds, shuffle=False, drop_last=False, **{k: v for k, v in common.items() if k != 'drop_last'}),
    }
