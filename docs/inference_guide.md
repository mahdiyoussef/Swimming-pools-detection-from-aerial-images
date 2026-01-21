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

### Tiled Inference (Large Images)

For massive aerial scenes (e.g., 10000x10000+ pixels), use the `--tiled` flag to process the image in sliding window tiles.

```bash
python inference/detect_pools.py \
    --input data/raw/kaggle/TEST_SET_ALPES_MARITIMES.3.png \
    --model weights/best.pt \
    --tiled \
    --tile-size 640 \
    --tile-overlap 128
```

This prevents GPU memory errors and ensures high-resolution detection of small pools across large regions.

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
Pool Segmentation Coordinates
Confidence: 0.9500
Shape: oval
Vertex Count: 14
Boundary Points:
123,456
130,460
...
```

The coordinates represent a variable number of **polygon vertices** (x,y) tracing the actual pool border. The `Shape` field classifies the geometry into `rectangular`, `oval`, or `irregular`.

### output_image.jpg

Original image with:
- Blue outline (BGR: 255, 0, 0)
- 3 pixel thickness
- Drawn around each detected pool

## Programmatic Usage

```python
from inference.detect_pools import PoolDetector
from inference.postprocessing import (
    extract_pool_contour,  # High-precision GrabCut contour
    draw_pool_mask,       # Colored polygons
    save_segmentation_coordinates, 
    classify_pool_shape
)

# Initialize detector
detector = PoolDetector(model_path="weights/best.pt")

# Run detection
image, detections = detector.detect("aerial_image.jpg")

# Process each detection
for i, det in enumerate(detections):
    bbox = det["bbox"]
    
    # Extract HIGH-PRECISION contour
    contour = extract_pool_contour(image, np.array(bbox))
    
    # Classify and Draw
    shape = classify_pool_shape(contour)
    image = draw_pool_mask(image, contour, color=(0, 0, 255))
    
    # Save
    save_segmentation_coordinates(contour, f"coords_{i}.txt", det["confidence"], shape)
```

## Performance Tips

### Speed Optimization

1. Use GPU: `--device 0`
2. Use smaller model: `yolo26n.pt`
3. Reduce image size: `--img-size 416`
4. Enable TensorRT (if available)

### Accuracy Optimization

1. Use larger model: `yolo26x.pt`
2. Increase image size: `--img-size 1024`
3. Lower confidence threshold: `--conf-threshold 0.1`
4. Tune IoU threshold for your use case

## Benchmarks

| Model | Image Size | GPU | Inference Time |
|-------|------------|-----|----------------|
| yolo26n | 640 | RTX 3080 | ~5ms |
| yolo26s | 640 | RTX 3080 | ~7ms |
| yolo26m | 640 | RTX 3080 | ~12ms |
| yolo26l | 640 | RTX 3080 | ~18ms |
| yolo26x | 640 | RTX 3080 | ~30ms |

*Benchmarks are approximate and depend on hardware*
