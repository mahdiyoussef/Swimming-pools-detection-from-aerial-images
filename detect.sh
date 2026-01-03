#!/bin/bash
# Swimming Pool Detection - Run Detection on Test Images
# This script processes all images in test_images folder

set -e

echo "============================================"
echo "  Swimming Pool Detection"
echo "============================================"

# Configuration
INPUT_DIR="test_images"
OUTPUT_DIR="output"
MODEL="weights/best.pt"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: $INPUT_DIR folder not found"
    echo "Please create a 'test_images' folder and add images to it"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    echo "Please ensure the trained model exists at weights/best.pt"
    exit 1
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Count images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
echo "Found $IMAGE_COUNT images in $INPUT_DIR"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "No images found. Add .jpg, .jpeg, or .png files to $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run detection
echo "Running detection..."
python inference/detect_pools.py \
    --input "$INPUT_DIR" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  Detection Complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
