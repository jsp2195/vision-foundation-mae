# minimae-vit

Production-minded, config-driven Vision Transformer + Masked Autoencoder pipeline for self-supervised pretraining and downstream transfer.

## Features
- MAE pretraining on open datasets (CIFAR-10 default; STL-10 optional)
- Linear probe and full finetuning on CIFAR-10
- Encoder export and folder inference CLI
- Hydra config system with CLI overrides
- JSONL + TensorBoard logging, optional W&B
- Reproducibility manifest (resolved config, env, git hash)
- Pytest coverage for key invariants

## Quickstart
```bash
pip install -e .
python scripts/train_mae.py
python scripts/eval_linear_probe.py pretrained_ckpt=outputs/latest/checkpoints/last.pt
python scripts/train_finetune.py pretrained_ckpt=outputs/latest/checkpoints/last.pt
python scripts/export_encoder.py pretrained_ckpt=outputs/latest/checkpoints/last.pt
python scripts/infer.py classifier_ckpt=outputs/latest/checkpoints/best.pt image_dir=./some_images
```

## Structure
See `configs/`, `src/minimae_vit/`, `scripts/`, and `tests/`.

## Notes
- Default image size is 64x64 (CIFAR-10 upsampled from 32x32).
- MAE reconstruction visualization uses pixel-space unnormalization from stored per-patch statistics.
