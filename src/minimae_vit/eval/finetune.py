from pathlib import Path

from minimae_vit.train.loops import run_classifier


def run_finetune(cfg, out_dir: Path) -> float:
    return run_classifier(cfg, out_dir, finetune=True)
