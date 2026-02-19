from pathlib import Path

from minimae_vit.train.loops import run_classifier


def evaluate_linear_probe(cfg, out_dir: Path) -> float:
    return run_classifier(cfg, out_dir, finetune=False)
