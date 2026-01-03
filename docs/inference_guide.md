# Inference Guide

## Quick Start

```bash
python inference/detect_pools.py \
    --input path/to/image.jpg \
    --model logs/checkpoints/best.pt
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | str | required | Input image or directory |
| `--model` | str | required | Path to model weights |
| `--output-dir` | str | `output` | Output directory |
| `--conf-threshold` | float | `0.25` | Confidence threshold |
| `--iou-threshold` | float | `0.45` | NMS IoU threshold |
| `--img-size` | int | `640` | Inference image size |
| `--device` | str | `0` | CUDA device or `cpu` |

## Usage Examples

### Single Image

```bash
python inference/detect_pools.py \
    --input aerial_photo.jpg \
    --model weights/best.pt \
    --output-dir results
```

Output:
- `results/coordinates.txt`
- `results/output_image.jpg`

### Batch Processing

```bash
python inference/detect_pools.py \
    --input images_folder/ \
    --model weights/best.pt \
    --output-dir results
```

Output structure:
```
results/
├── image1/
│   ├── coordinates.txt
│   └── output_image.jpg
├── image2/
│   ├── coordinates.txt
│   └── output_image.jpg
...
```

### High Precision

```bash
python inference/detect_pools.py \
    --input image.jpg \
    --model weights/best.pt \
    --conf-threshold 0.5 \
    --iou-threshold 0.6
```

### CPU Inference

```bash
python inference/detect_pools.py \
    --input image.jpg \
    --model weights/best.pt \
    --device cpu
```

## Output Format

### coordinates.txt

```
Pool Detection Coordinates
Confidence: 0.9500
Boundary Points:
123,456
789,456
789,234
123,234
```

The coordinates represent the four corners of the bounding box:
1. Top-left (x1, y1)
2. Top-right (x2, y1)
3. Bottom-right (x2, y2)
4. Bottom-left (x1, y2)

### output_image.jpg

Original image with:
- Blue outline (BGR: 255, 0, 0)
- 3 pixel thickness
- Drawn around each detected pool

## Programmatic Usage

```python
from inference.detect_pools import PoolDetector
from inference.postprocessing import (
    extract_boundary_coordinates,
    draw_pool_outline,
    save_coordinates
)

# Initialize detector
detector = PoolDetector(
    model_path="weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    device="0"
)

# Run detection
image, detections = detector.detect("aerial_image.jpg")

# Process each detection
for det in detections:
    bbox = det["bbox"]
    confidence = det["confidence"]
    
    # Get coordinates
    coords = extract_boundary_coordinates(
        np.array(bbox),
        image.shape[:2]
    )
    
    # Draw outline
    image = draw_pool_outline(image, coords)
    
    # Save coordinates
    save_coordinates(coords, "coordinates.txt", confidence)
```

## Performance Tips

### Speed Optimization

1. Use GPU: `--device 0`
2. Use smaller model: `yolov11n.pt`
3. Reduce image size: `--img-size 416`
4. Enable TensorRT (if available)

### Accuracy Optimization

1. Use larger model: `yolov11x.pt`
2. Increase image size: `--img-size 1024`
3. Lower confidence threshold: `--conf-threshold 0.1`
4. Tune IoU threshold for your use case

## Benchmarks

| Model | Image Size | GPU | Inference Time |
|-------|------------|-----|----------------|
| yolov11n | 640 | RTX 3080 | ~5ms |
| yolov11s | 640 | RTX 3080 | ~7ms |
| yolov11m | 640 | RTX 3080 | ~12ms |
| yolov11l | 640 | RTX 3080 | ~18ms |
| yolov11x | 640 | RTX 3080 | ~30ms |

*Benchmarks are approximate and depend on hardware*
