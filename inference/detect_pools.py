"""
Pool Detection Script for Swimming Pool Detection System.

This module handles inference on images using trained YOLOv11 models,
generating coordinates.txt and output_image.jpg files.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class PoolDetector:
    """
    Swimming pool detection using YOLOv11.

    Attributes:
        model: YOLOv11 model instance.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IOU threshold for NMS.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "0"
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = self._load_model()

    def _load_model(self) -> "YOLO":
        from ultralytics import YOLO
        logger.info(f"Loading model from: {self.model_path}")
        model = YOLO(str(self.model_path))
        return model

    def detect(
        self,
        image_path: str,
        image_size: int = 640
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run detection on a single image.

        Args:
            image_path: Path to the input image.
            image_size: Inference image size.

        Returns:
            Tuple of (original_image, list of detections).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Run inference
        start_time = time.time()
        
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time * 1000:.2f} ms")
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Get bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detection = {
                    "bbox": xyxy.tolist(),  # [x1, y1, x2, y2]
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": "swimming_pool"
                }
                detections.append(detection)
        
        logger.info(f"Detected {len(detections)} swimming pool(s)")
        
        return original_image, detections

    def detect_batch(
        self,
        image_dir: str,
        image_size: int = 640
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection on all images in a directory.

        Args:
            image_dir: Directory containing images.
            image_size: Inference image size.

        Returns:
            Dictionary mapping image paths to their detections.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Not a directory: {image_dir}")
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        all_detections = {}
        
        for image_path in image_files:
            try:
                _, detections = self.detect(str(image_path), image_size)
                all_detections[str(image_path)] = detections
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
        
        return all_detections


def process_single_image(
    detector: PoolDetector,
    image_path: str,
    output_dir: str,
    image_size: int = 640
) -> None:
    """
    Process a single image and generate outputs.

    Args:
        detector: PoolDetector instance.
        image_path: Path to input image.
        output_dir: Directory for output files.
        image_size: Inference image size.
    """
    from inference.postprocessing import (
        draw_pool_outline,
        extract_boundary_coordinates,
        save_coordinates
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run detection
    original_image, detections = detector.detect(image_path, image_size)
    
    # Process each detection
    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        
        # Extract coordinates
        coordinates = extract_boundary_coordinates(
            np.array(bbox),
            original_image.shape[:2]
        )
        
        # Draw outline on image
        original_image = draw_pool_outline(
            original_image,
            coordinates,
            color=(0, 0, 255),  # Red in BGR
            thickness=3
        )
        
        # Save coordinates
        suffix = f"_{i}" if len(detections) > 1 else ""
        coord_path = output_dir / f"coordinates{suffix}.txt"
        save_coordinates(coordinates, str(coord_path), confidence)
    
    # Save output image
    output_image_path = output_dir / "output_image.jpg"
    cv2.imwrite(str(output_image_path), original_image)
    logger.info(f"Saved output image: {output_image_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect swimming pools in aerial images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IOU threshold for NMS"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device or 'cpu'"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for detection."""
    args = parse_arguments()
    
    # Initialize detector
    detector = PoolDetector(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        process_single_image(
            detector,
            str(input_path),
            args.output_dir,
            args.img_size
        )
    elif input_path.is_dir():
        # Directory of images
        for img_file in input_path.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img_output_dir = Path(args.output_dir) / img_file.stem
                try:
                    process_single_image(
                        detector,
                        str(img_file),
                        str(img_output_dir),
                        args.img_size
                    )
                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
