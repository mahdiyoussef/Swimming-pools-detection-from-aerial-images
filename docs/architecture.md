# System Architecture

## Overview

The Swimming Pool Detection System uses YOLOv11 for detecting swimming pools in aerial imagery. The architecture is designed for production use with modular components.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Swimming Pool Detection System                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Dataset    │───▶│ Preprocessing │───▶│   Training   │       │
│  │  Management  │    │   Pipeline    │    │   Pipeline   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Kaggle/RF   │    │ Augmentation │    │   YOLOv11    │       │
│  │     APIs     │    │  & Splits    │    │    Model     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                            ┌──────────────────────────────┐     │
│                            │      Inference Pipeline       │     │
│                            ├──────────────────────────────┤     │
│                            │  Detection ─▶ Postprocessing  │     │
│                            │      ▼              ▼         │     │
│                            │  coordinates.txt  output.jpg  │     │
│                            └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline

1. **Download** - Fetch datasets from Kaggle/Roboflow
2. **Preprocess** - Convert annotations to YOLO format
3. **Augment** - Apply data augmentation
4. **Split** - Create train/val/test splits (70/20/10)
5. **Train** - Train YOLOv11 model
6. **Validate** - Compute metrics (mAP, precision, recall)

### Inference Pipeline

1. **Load** - Load trained model weights
2. **Preprocess** - Resize and normalize input image
3. **Detect** - Run YOLOv11 inference
4. **Postprocess** - Extract coordinates, apply NMS
5. **Output** - Generate coordinates.txt and output_image.jpg

## YOLOv11 Model Architecture

YOLOv11 builds on YOLO evolution with:

- **Backbone**: CSPDarknet with attention mechanisms
- **Neck**: PANet for multi-scale feature fusion
- **Head**: Decoupled detection head

### Model Variants

| Variant | Depth | Width | Architecture Notes |
|---------|-------|-------|-------------------|
| Nano    | 0.33  | 0.25  | Mobile deployment |
| Small   | 0.33  | 0.50  | Edge devices |
| Medium  | 0.67  | 0.75  | Balanced |
| Large   | 1.00  | 1.00  | High accuracy |
| X-Large | 1.00  | 1.25  | Maximum accuracy |

## Custom Components

### Attention Mechanisms

- **CBAM** - Channel and spatial attention
- **SE Blocks** - Squeeze-and-excitation

### Loss Functions

- **CIoU Loss** - Complete IoU for box regression
- **Focal Loss** - Handle class imbalance

## Configuration System

All configurations use YAML format:

- `dataset_config.yaml` - Data paths, splits, augmentation
- `training_config.yaml` - Model, optimizer, scheduler

## Logging and Monitoring

- **TensorBoard** - Training metrics visualization
- **File Logging** - Detailed execution logs
- **Checkpointing** - Model state preservation
