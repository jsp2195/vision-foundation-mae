# minimae-vit

Production-ready, config-driven Vision Transformer MAE training and transfer workflow.

## Features
- MAE self-supervised pretraining on CIFAR-10 (or STL-10 unlabeled)
- Linear probe and full finetune evaluation
- Encoder export and inference CLI
- Hydra configs, TensorBoard + JSONL logging, optional W&B
- Deterministic seed controls and run manifest capture
- Pytest coverage for core invariants

## Install
```bash
pip install -e .
```

## Quickstart
```bash
python scripts/train_mae.py
python scripts/eval_linear_probe.py
python scripts/train_finetune.py
python scripts/export_encoder.py ckpt_path=outputs/latest/checkpoints/last.pt
python scripts/infer.py ckpt_path=outputs/latest/checkpoints/best.pt image_dir=./some_images
```

All run artifacts are written under `outputs/<run_id>/`.

## Tests
```bash
pytest -q
```
