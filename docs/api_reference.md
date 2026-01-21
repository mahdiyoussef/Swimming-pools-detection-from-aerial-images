# API Reference

## Dataset Management

### DatasetDownloader

```python
from datasets.download_dataset import DatasetDownloader

downloader = DatasetDownloader(config_path="config/dataset_config.yaml")
```

**Methods:**

#### `download_from_kaggle(dataset_name: str = None) -> Path`
Download dataset from Kaggle.

#### `download_from_roboflow(workspace: str, project: str, version: int) -> Path`
Download dataset from Roboflow.

#### `verify_integrity(dataset_path: Path) -> bool`
Verify dataset has images and labels.

#### `organize_files(source_path: Path, target_path: Path = None) -> None`
Organize files into standard structure.

---

## Preprocessing

### DatasetSplitter

```python
from preprocessing.create_splits import DatasetSplitter

splitter = DatasetSplitter(config_path="config/dataset_config.yaml")
splitter.create_splits()
```

**Methods:**

#### `find_image_label_pairs(images_dir, labels_dir) -> List[Tuple[Path, Path]]`
Find matching image-label pairs.

#### `split_data(pairs) -> Tuple[List, List, List]`
Split into train/val/test sets.

#### `generate_dataset_yaml(output_path: Path) -> None`
Generate YOLO26 dataset.yaml.

---

### AugmentationPipeline

```python
from preprocessing.augmentation import AugmentationPipeline

pipeline = AugmentationPipeline(config_path="config/dataset_config.yaml")
aug_img, aug_bboxes, aug_labels = pipeline.apply(image, bboxes, class_labels)
```

**Methods:**

#### `apply(image, bboxes, class_labels) -> Tuple[np.ndarray, List, List]`
Apply augmentation to image and bounding boxes.

#### `get_training_transform(image_size: int = 640) -> A.Compose`
Get training augmentation pipeline.

#### `get_validation_transform(image_size: int = 640) -> A.Compose`
Get validation transform (resize only).

---

### AnnotationConverter

```python
from preprocessing.annotation_converter import AnnotationConverter

converter = AnnotationConverter(class_mapping={"swimming_pool": 0})
```

**Methods:**

#### `coco_to_yolo(coco_json_path, output_dir) -> int`
Convert COCO JSON to YOLO format.

#### `voc_to_yolo(xml_path, output_path) -> int`
Convert Pascal VOC XML to YOLO format.

#### `polygon_to_bbox(polygon: List[Tuple]) -> Tuple[float, float, float, float]`
Convert polygon to bounding box.

---

## Model Configuration

### YOLO26Config

```python
from models.yolo26_config import YOLO26Config

config = YOLO26Config(
    variant="yolo26m",
    num_classes=1,
    pretrained=True
)
model = config.create_model()
```

**Methods:**

#### `create_model() -> YOLO`
Create YOLO26 model instance.

#### `get_training_args(**kwargs) -> Dict[str, Any]`
Get training arguments for model.train().

#### `get_inference_args(**kwargs) -> Dict[str, Any]`
Get inference arguments for model.predict().

#### `list_variants() -> None`
Print available model variants.

---

## Training

### Trainer

```python
from training.train import Trainer

trainer = Trainer(config_path="config/training_config.yaml")
trainer.train(epochs=100, batch_size=16)
```

**Methods:**

#### `train(model_variant, epochs, batch_size, resume, **kwargs) -> None`
Execute training.

---

### Validator

```python
from training.validate import Validator

validator = Validator(
    model_path="weights/best.pt",
    data_yaml="data/dataset.yaml"
)
metrics = validator.validate(split="val")
```

**Methods:**

#### `validate(split, conf_threshold, iou_threshold, ...) -> Dict[str, Any]`
Run validation and compute metrics.

---

## Inference

### PoolDetector

```python
from inference.detect_pools import PoolDetector

detector = PoolDetector(
    model_path="weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45
)
image, detections = detector.detect("image.jpg")
```

**Methods:**

#### `detect(image_path, image_size) -> Tuple[np.ndarray, List[Dict]]`
Run detection on single image.

#### `detect_tiled(image_path, tile_size, tile_overlap, image_size) -> Tuple[np.ndarray, List[Dict]]`
Run tiled/sliding window detection on massive images (10000x10000+).

#### `detect_batch(image_dir, image_size) -> Dict[str, List[Dict]]`
Run detection on directory of images.

---

#### `extract_pool_contour(image, bbox, epsilon_factor, min_area_ratio) -> List[Tuple[int, int]]`
High-precision GrabCut-based contour extraction for accurate boundaries.

#### `draw_pool_mask(image, polygon, color, alpha, outline_thickness) -> np.ndarray`
Draw semi-transparent filled mask with high-visibility outline (e.g. Red).

#### `classify_pool_shape(polygon) -> str`
Classify geometry into `rectangular`, `oval`, `circular`, or `irregular`.

#### `save_segmentation_coordinates(polygon, output_path, confidence, shape) -> None`
Save extended polygon data with shape classification to text file.

#### `extract_boundary_coordinates(bbox, img_shape) -> List[Tuple[int, int]]`
Convert bbox to corner coordinates (fallback).

---

## Utility Functions

```python
from inference.utils import (
    load_image,
    save_image,
    resize_image,
    visualize_detections
)
```

#### `load_image(image_path, color_mode) -> np.ndarray`
Load image from disk.

#### `save_image(image, output_path, quality) -> None`
Save image to disk.

#### `resize_image(image, size, keep_aspect_ratio) -> Tuple[np.ndarray, Tuple]`
Resize image with optional aspect ratio preservation.

#### `visualize_detections(image, detections, color, thickness) -> np.ndarray`
Draw all detections on image.
