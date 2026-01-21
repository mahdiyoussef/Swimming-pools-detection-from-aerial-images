# Training Guide

## Prerequisites

Before training, ensure you have:

1. Completed setup (`./scripts/setup.sh` or `scripts\setup.bat`)
2. Downloaded and prepared dataset
3. GPU with sufficient VRAM (8GB+ recommended)

## Dataset Preparation

### 1. Download Dataset

```bash
# From Kaggle
python datasets/download_dataset.py --source kaggle

# From Roboflow
export ROBOFLOW_API_KEY=your_api_key
python datasets/download_dataset.py --source roboflow
```

### 2. Create Splits

```bash
python preprocessing/create_splits.py
```

This creates:
- `data/splits/train/` - 70% of data
- `data/splits/val/` - 20% of data
- `data/splits/test/` - 10% of data
- `data/dataset.yaml` - YOLO26 dataset config

## Training Configuration

Edit `config/training_config.yaml`:

```yaml
model:
  variant: yolo26m      # Choose: n, s, m, l, x
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  image_size: 640

optimizer:
  name: AdamW
  learning_rate: 0.001

early_stopping:
  patience: 20
  monitor: val/mAP50
```

## Running Training

### Basic Training

```bash
python training/train.py --config config/training_config.yaml
```

### Advanced Options

```bash
python training/train.py \
    --config config/training_config.yaml \
    --model yolo26l \
    --epochs 200 \
    --batch-size 8 \
    --img-size 640 \
    --device 0
```

### Multi-GPU Training

```bash
python training/train.py \
    --config config/training_config.yaml \
    --device 0,1
```

### Resume Training

```bash
python training/train.py \
    --config config/training_config.yaml \
    --resume logs/checkpoints/yolo26m_20260102_180000/weights/last.pt
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Metrics tracked:
- `train/loss` - Total training loss
- `train/box_loss` - Bounding box loss
- `train/cls_loss` - Classification loss
- `val/mAP50` - Validation mAP@0.5
- `val/mAP50-95` - Validation mAP@0.5:0.95
- `lr/learning_rate` - Current learning rate

### Log Files

Training logs saved to: `logs/training/train_<timestamp>.log`

## Checkpoints

Saved to: `logs/checkpoints/<run_name>/weights/`

- `best.pt` - Best model by validation mAP
- `last.pt` - Latest checkpoint
- `checkpoint_epoch_X_mAP_Y.pt` - Periodic saves

## Hyperparameter Tuning

### Learning Rate

Start with 0.001, decrease if training is unstable:

```yaml
optimizer:
  learning_rate: 0.0001  # Lower for fine-tuning
```

### Batch Size

Larger = faster, but requires more VRAM:

- 4GB VRAM: batch_size 4-8
- 8GB VRAM: batch_size 8-16
- 16GB VRAM: batch_size 16-32

### Image Size

Larger = better accuracy, slower training:

- 416: Fast, lower accuracy
- 640: Balanced (recommended)
- 1024: High accuracy, slow

## Data Augmentation

Enabled in `config/dataset_config.yaml`:

```yaml
augmentation:
  enabled: true
  horizontal_flip: true
  vertical_flip: true
  rotation_limit: 15
  brightness_limit: 0.2
  contrast_limit: 0.2
```

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Use smaller model
--model yolo26n
```

### Slow Training

- Enable AMP (automatic mixed precision)
- Use more workers: increase `training.workers`
- Use SSD instead of HDD for data

### Poor Convergence

- Increase epochs
- Lower learning rate
- Add more augmentation
- Check annotation quality
