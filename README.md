# Swimming Pool Detection & Segmentation

Automatic detection and **high-precision instance segmentation** of swimming pools in aerial/satellite imagery using YOLO and GrabCut refinement.

## Project Description

This project detects and segments swimming pools of any shape (rectangular, oval, circular, irregular) in aerial images and outputs:
- **coordinates.txt**: High-fidelity polygon boundary coordinates with **shape classification**
- **output_image.jpg**: Input image with bold red outlines tracing the **exact pool borders**

The system uses a hybrid approach: **YOLO** for robust detection and **GrabCut + Color Segmentation** for precise polygon contours, achieving accurate shape-aware results even without a segmented training dataset.

---

## Quick Start

### 1. Setup

```bash
# Clone and enter project
cd swimming-pool-detection

# Run setup script (creates venv and installs dependencies)
./setup

# Activate virtual environment
source venv/bin/activate
```

### 2. Detect Swimming Pools

**Option A: Use the detection script**
```bash
# Put your images in test_images folder
cp your_aerial_image.jpg test_images/

# Run detection
./detect.sh
```

**Option B: Run directly with Python**
```bash
# Single image
python inference/detect_pools.py --input path/to/image.jpg

# Directory of images
python inference/detect_pools.py --input path/to/images/ --output-dir results
```

### 3. View Results

Results are saved to the `output/` folder:
```
output/
  image_name/
    coordinates.txt      # Pool boundary coordinates
    output_image.jpg     # Image with red pool outlines
```

---

## Sample Results

Here are examples of the model detecting swimming pools in aerial imagery:

| Input Image | Detection Result |
|-------------|------------------|
| ![Sample 1](test_images/000000079.jpg) | ![Result 1](output/000000079/output_image.jpg) |
| ![Sample 2](test_images/000000136.jpg) | ![Result 2](output/000000136/output_image.jpg) |
| ![Sample 3](test_images/000000216.jpg) | ![Result 3](output/000000216/output_image.jpg) |
| ![Sample 4](test_images/000000292.jpg) | ![Result 4](output/000000292/output_image.jpg) |
| ![Sample 5](test_images/000000378.jpg) | ![Result 5](output/000000378/output_image.jpg) |

The model (YOLO26s) achieves **97.7% mAP50** and **93.6% precision** on the validation set. 

### Coordinates File Format

Each detected pool generates a detailed `coordinates.txt` file containing polygon vertices:

```
Pool Segmentation Coordinates
Confidence: 0.8106
Shape: oval
Vertex Count: 12
Boundary Points:
139,75
145,72
155,75
...
```

The coordinates represent the **actual boundary points** (x,y) of the pool water in pixels.

---

## Detection Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input image or directory |
| `--model` | `weights/best.pt` | Path to model weights |
| `--output-dir` | `output` | Output directory |
| `--conf-threshold` | `0.5` | Detection confidence threshold |
| `--img-size` | `640` | Inference image size |

**Example with custom options:**
```bash
python inference/detect_pools.py \
    --input aerial_photo.jpg \
    --conf-threshold 0.5 \
    --output-dir my_results
```

---

## Model Information

| Attribute | Value |
|-----------|-------|
| Architecture | YOLO26s |
| Input Size | 512x512 |
| Training Images | 856 |
| Validation Images | 244 |
| Test Images | 124 |
| **mAP50** | **0.977** |
| **mAP50-95** | **0.599** |
| Precision | 0.936 |
| Recall | 0.942 |
| Inference Speed | ~25ms/image (GPU) |

---

## Training (Optional)

To train your own model:

### 1. Prepare Dataset
```bash
# Download dataset from Kaggle
python datasets/download_dataset.py --source kaggle

# Convert annotations to YOLO format
python preprocessing/annotation_converter.py \
    --format voc \
    --input data/raw/kaggle/labels \
    --output data/processed/labels \
    --class-names pool

# Create train/val/test splits
python preprocessing/create_splits.py
```

### 2. Train Model
```bash
python training/train.py --config config/training_config.yaml
```

### 3. Training Configuration

Edit `config/training_config.yaml` to customize:
- Model variant (yolo26n/s/m/l/x)
- Epochs, batch size, image size
- Learning rate, optimizer
- Early stopping patience

---

## Project Structure

```
swimming-pool-detection/
├── detect.sh              # Quick detection script
├── weights/best.pt        # Trained model
├── inference/             # Detection scripts
├── training/              # Training scripts
├── preprocessing/         # Data preparation
├── config/                # Configuration files
├── data/                  # Dataset storage
├── test_images/           # Put images here for detection
└── output/                # Detection results
```

---

## Requirements

- Python 3.9+
- CUDA GPU (recommended) or CPU
- ~4GB GPU memory for inference

---

## Detailed Pipeline Documentation

This section provides comprehensive documentation of the complete pipeline from data preprocessing to model training and testing/inference.

---

### Step 1: Dataset Download

The system supports downloading datasets from **Kaggle** and **Roboflow**. The primary dataset used is `alexj21/swimming-pool-512x512` from Kaggle.

**Module:** `datasets/download_dataset.py`

**Functionality:**
- Authenticates with Kaggle API using credentials from `.env` file or `~/.kaggle/kaggle.json`
- Downloads and extracts dataset archives
- Verifies dataset integrity (checks for images and labels)
- Organizes files into a standard structure

**Command:**
```bash
# Download from Kaggle (default)
python datasets/download_dataset.py --source kaggle

# Download from specific Kaggle dataset
python datasets/download_dataset.py --source kaggle --kaggle-dataset owner/dataset-name

# Download from all enabled sources
python datasets/download_dataset.py --source all
```

**Configuration:** `config/dataset_config.yaml`
```yaml
sources:
  kaggle:
    dataset_name: "alexj21/swimming-pool-512x512"
    enabled: true
  roboflow:
    workspace: "detection-opefo"
    project: "swimming_pool"
    version: 1
    enabled: false
```

**Output Structure:**
```
data/
├── raw/kaggle/          # Raw downloaded files
├── processed/
│   ├── images/          # Organized image files
│   └── labels/          # Organized YOLO label files
```

---

### Step 2: Annotation Conversion

Converts annotations from various formats (COCO JSON, Pascal VOC XML) to YOLO format.

**Module:** `preprocessing/annotation_converter.py`

**Functionality:**
- **COCO to YOLO:** Converts COCO JSON annotations with [x_min, y_min, width, height] format
- **VOC to YOLO:** Converts Pascal VOC XML with [xmin, ymin, xmax, ymax] format
- **Polygon to Bbox:** Converts polygon coordinates to bounding boxes
- **Validation:** Verifies YOLO annotation format correctness

**YOLO Format:**
Each line in a `.txt` label file:
```
<class_id> <x_center> <y_center> <width> <height>
```
- All values are **normalized** to [0, 1] relative to image dimensions
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Box dimensions

**Command:**
```bash
# Convert from Pascal VOC format
python preprocessing/annotation_converter.py \
    --format voc \
    --input data/raw/kaggle/labels \
    --output data/processed/labels \
    --class-names swimming_pool

# Convert from COCO format
python preprocessing/annotation_converter.py \
    --format coco \
    --input data/raw/annotations.json \
    --output data/processed/labels
```

**Key Functions:**
| Function | Description |
|----------|-------------|
| `coco_to_yolo()` | Convert COCO JSON to YOLO format |
| `voc_to_yolo()` | Convert single VOC XML to YOLO |
| `voc_dir_to_yolo()` | Convert directory of VOC XMLs |
| `polygon_to_bbox()` | Convert polygon to bounding box |
| `validate_yolo_annotation()` | Validate YOLO label file |

---

### Step 3: Data Augmentation

Applies various augmentation techniques to increase dataset diversity and improve model generalization.

**Module:** `preprocessing/augmentation.py`

**Augmentation Pipeline (using Albumentations):**

| Augmentation | Description | Default Probability |
|--------------|-------------|---------------------|
| Horizontal Flip | Mirror image horizontally | 0.5 |
| Vertical Flip | Mirror image vertically | 0.5 |
| Rotation | Random rotation (-15deg to +15deg) | 0.5 |
| Brightness/Contrast | Random brightness and contrast adjustment | 0.5 |
| Random Scale | Scale image (1 +/- 0.2) | 0.5 |
| Gaussian Noise | Add random noise | 0.2 |
| Gaussian Blur | Apply blur (kernel 3-5) | 0.2 |
| HSV Adjustment | Hue, saturation, value shifts | 0.3 |
| CLAHE | Contrast Limited Adaptive Histogram Equalization | 0.2 |

**Command:**
```bash
python preprocessing/augmentation.py \
    --images-dir data/processed/images \
    --labels-dir data/processed/labels \
    --output-images data/augmented/images \
    --output-labels data/augmented/labels \
    --num-augmentations 3
```

**Configuration:** `config/dataset_config.yaml`
```yaml
augmentation:
  enabled: true
  horizontal_flip: true
  vertical_flip: true
  rotation_limit: 15
  brightness_limit: 0.2
  contrast_limit: 0.2
  scale_limit: 0.2
```

**Note:** Bounding boxes are automatically transformed along with images, maintaining annotation accuracy through the augmentation process.

---

### Step 4: Train/Validation/Test Split

Splits the dataset into train, validation, and test sets while ensuring no data leakage.

**Module:** `preprocessing/create_splits.py`

**Default Split Ratios:**
| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 20% | Hyperparameter tuning, early stopping |
| Test | 10% | Final model evaluation |

**Features:**
- **Random shuffling** with configurable seed for reproducibility
- **Stratified splitting** (optional) based on class distribution
- **Data leakage verification** ensures no overlap between splits
- **Automatic dataset.yaml generation** for YOLO26 compatibility

**Command:**
```bash
# Using default configuration
python preprocessing/create_splits.py

# With custom configuration
python preprocessing/create_splits.py --config config/dataset_config.yaml

# With custom directories
python preprocessing/create_splits.py \
    --images-dir data/processed/images \
    --labels-dir data/processed/labels
```

**Output Structure:**
```
data/
├── splits/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── dataset.yaml
```

**Generated `dataset.yaml`:**
```yaml
path: /path/to/project/data
train: splits/train/images
val: splits/val/images
test: splits/test/images
nc: 1
names: ['swimming_pool']
```

---

### Step 5: Model Training

Trains the YOLO26 model on the prepared dataset.

**Module:** `training/train.py`

**Training Configuration:** `config/training_config.yaml`

**Key Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLO26s | Small variant (balanced speed/accuracy) |
| Epochs | 200 | Maximum training epochs |
| Batch Size | 2 | Samples per batch (adjusted for GPU memory) |
| Image Size | 512x512 | Input image resolution |
| Learning Rate | 0.001 | Initial learning rate |
| Optimizer | AdamW | Optimizer with weight decay |
| Early Stopping | 50 epochs | Patience for early stopping |

**Advanced Augmentations (during training):**
| Augmentation | Value | Description |
|--------------|-------|-------------|
| Mosaic | 1.0 | Combines 4 images |
| MixUp | 0.1 | Blends two images |
| Copy-Paste | 0.1 | Pastes objects between images |
| HSV-H | 0.015 | Hue augmentation |
| HSV-S | 0.7 | Saturation augmentation |
| HSV-V | 0.4 | Value augmentation |
| Rotation | 10.0 | Rotation degrees |
| Scale | 0.5 | Scale factor |
| Flip LR | 0.5 | Horizontal flip probability |
| Flip UD | 0.1 | Vertical flip probability |

**Command:**
```bash
# Basic training
python training/train.py --config config/training_config.yaml

# With overrides
python training/train.py \
    --config config/training_config.yaml \
    --model yolo26s \
    --epochs 100 \
    --batch-size 4 \
    --img-size 512 \
    --device 0

# Resume from checkpoint
python training/train.py \
    --config config/training_config.yaml \
    --resume logs/checkpoints/yolo26s_YYYYMMDD_HHMMSS/weights/last.pt
```

**Training Callbacks:**
- **Early Stopping:** Stops training if validation mAP50 doesn't improve for 50 epochs
- **Model Checkpointing:** Saves best model (highest mAP50) and periodic checkpoints
- **Learning Rate Scheduling:** Cosine annealing schedule
- **TensorBoard Logging:** Real-time training visualization

**Output:**
```
logs/checkpoints/<run_name>/
├── weights/
│   ├── best.pt          # Best model (highest mAP50)
│   └── last.pt          # Latest checkpoint
├── results.csv          # Training metrics per epoch
├── confusion_matrix.png # Confusion matrix visualization
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
└── results.png          # Training plots
```

---

### Step 6: Model Validation

Evaluates the trained model on validation or test sets.

**Module:** `training/validate.py`

**Metrics Computed:**
| Metric | Description |
|--------|-------------|
| mAP50 | Mean Average Precision at IoU 0.5 |
| mAP50-95 | Mean AP averaged over IoU 0.5 to 0.95 |
| Precision | True Positives / (True Positives + False Positives) |
| Recall | True Positives / (True Positives + False Negatives) |
| F1 Score | Harmonic mean of Precision and Recall |

**Command:**
```bash
# Validate on validation set
python training/validate.py \
    --model weights/best.pt \
    --data data/dataset.yaml \
    --split val

# Validate on test set
python training/validate.py \
    --model weights/best.pt \
    --data data/dataset.yaml \
    --split test \
    --batch-size 16 \
    --device 0
```

**Validation Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf-threshold` | 0.001 | Confidence threshold for predictions |
| `--iou-threshold` | 0.7 | IoU threshold for NMS |
| `--batch-size` | 16 | Batch size for validation |
| `--device` | 0 | CUDA device or 'cpu' |

**Output:**
Saves JSON metrics file to `logs/validation/validation_<split>_<timestamp>.json`

---

### Step 7: Inference/Testing

Runs detection on new images and generates output files.

**Module:** `inference/detect_pools.py`

**Inference Pipeline:**
1. Load trained model (`weights/best.pt`)
2. Read input image(s)
3. Preprocess (resize to inference size)
4. Run YOLO26 inference
5. Apply NMS (Non-Maximum Suppression)
6. Extract bounding box coordinates
7. Draw red outlines on detected pools
8. Save `coordinates.txt` and `output_image.jpg`

**Command Options:**
```bash
# Single image
python inference/detect_pools.py --input path/to/image.jpg

# Directory of images
python inference/detect_pools.py --input path/to/images/

# With custom options
python inference/detect_pools.py \
    --input test_images/ \
    --model weights/best.pt \
    --output-dir results/ \
    --conf-threshold 0.5 \
    --iou-threshold 0.45 \
    --img-size 640 \
    --device 0
```

**Quick Detection Script:**
```bash
# Put images in test_images/ folder, then run:
./detect.sh
```

**Output Files:**
```
output/<image_name>/
├── coordinates.txt      # Pool boundary coordinates
└── output_image.jpg     # Image with red pool outlines
```

**Coordinates File Format:**
```
Pool Detection Coordinates
Confidence: 0.8106
Boundary Points:
139,75
175,75
175,117
139,117
```

**Postprocessing Functions:** `inference/postprocessing.py`
| Function | Description |
|----------|-------------|
| `extract_boundary_coordinates()` | Convert bbox to corner points |
| `draw_pool_outline()` | Draw polygon on image |
| `save_coordinates()` | Write coordinates to file |
| `calculate_pool_area()` | Calculate pool area in pixels |
| `calculate_pool_dimensions()` | Get width and height |

---

### Step 8: Running Tests

Comprehensive test suite for all modules.

**Test Modules:** `tests/`
- `test_preprocessing.py` - Dataset splitting, annotation conversion, augmentation
- `test_inference.py` - Detection output format, image processing, file generation
- `test_postprocessing.py` - Coordinate extraction, drawing functions
- `test_data_loading.py` - Data loading utilities

**Command:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_inference.py -v

# Run specific test class
pytest tests/test_preprocessing.py::TestAnnotationConverter -v
```

**Test Coverage Areas:**
| Module | Tests |
|--------|-------|
| Preprocessing | Split ratios, YOLO format, bbox normalization |
| Inference | Detection output format, image loading, coordinates |
| Postprocessing | Boundary extraction, outline drawing |

---

### Complete Pipeline Example

```bash
# 1. Setup environment
./setup
source venv/bin/activate

# 2. Download dataset
python datasets/download_dataset.py --source kaggle

# 3. Convert annotations (if needed)
python preprocessing/annotation_converter.py \
    --format voc \
    --input data/raw/kaggle/labels \
    --output data/processed/labels \
    --class-names swimming_pool

# 4. Create train/val/test splits
python preprocessing/create_splits.py

# 5. Train model
python training/train.py --config config/training_config.yaml

# 6. Validate on test set
python training/validate.py \
    --model logs/checkpoints/<run>/weights/best.pt \
    --split test

# 7. Copy best model to weights folder
cp logs/checkpoints/<run>/weights/best.pt weights/best.pt

# 8. Run inference on new images
python inference/detect_pools.py \
    --input test_images/ \
    --output-dir output/
```

---

### Configuration Files Reference

| File | Purpose |
|------|---------|
| `config/dataset_config.yaml` | Dataset sources, splits, augmentation settings |
| `config/training_config.yaml` | Model, optimizer, training hyperparameters |
| `data/dataset.yaml` | YOLO26 dataset configuration (auto-generated) |
| `requirements.txt` | Python dependencies |
| `.env` | Kaggle/Roboflow API credentials |

---

## License

MIT License
