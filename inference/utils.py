"""
Utility Functions for Swimming Pool Detection System.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    color_mode: str = "bgr"
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file.
        color_mode: Color mode ('bgr', 'rgb', 'gray').

    Returns:
        Image as numpy array.

    Raises:
        FileNotFoundError: If image doesn't exist.
        ValueError: If image loading fails.
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array.
        output_path: Path to save the image.
        quality: JPEG quality (0-100).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    cv2.imwrite(str(output_path), image, params)
    
    logger.debug(f"Saved image: {output_path}")


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize an image.

    Args:
        image: Input image.
        size: Target size (width, height).
        keep_aspect_ratio: Whether to maintain aspect ratio.

    Returns:
        Tuple of (resized image, scale factors (sx, sy)).
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    if keep_aspect_ratio:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = target_w, target_h
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    scale_x = new_w / w
    scale_y = new_h / h
    
    return resized, (scale_x, scale_y)


def pad_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to target size.

    Args:
        image: Input image.
        target_size: Target size (width, height).
        pad_value: Padding value (grayscale).

    Returns:
        Tuple of (padded image, padding offset (dx, dy)).
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate padding
    pad_x = (target_w - w) // 2
    pad_y = (target_h - h) // 2
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=np.uint8)
    else:
        padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    
    # Place image in center
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = image
    
    return padded, (pad_x, pad_y)


def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Get image information.

    Args:
        image_path: Path to the image.

    Returns:
        Dictionary with image information.
    """
    image = load_image(image_path)
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        "path": str(image_path),
        "width": w,
        "height": h,
        "channels": channels,
        "size_bytes": Path(image_path).stat().st_size
    }


def is_valid_image(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid image.

    Args:
        file_path: Path to the file.

    Returns:
        True if file is a valid image.
    """
    try:
        image = cv2.imread(str(file_path))
        return image is not None
    except Exception:
        return False


def list_images(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    List all image files in a directory.

    Args:
        directory: Directory path.
        extensions: List of valid extensions (default: common image formats).

    Returns:
        List of image file paths.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    directory = Path(directory)
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def visualize_detections(
    image: np.ndarray,
    detections: List[dict],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
    show_labels: bool = True
) -> np.ndarray:
    """
    Visualize detections on an image.

    Args:
        image: Input image.
        detections: List of detection dictionaries.
        color: BGR color for boxes.
        thickness: Line thickness.
        show_labels: Whether to show confidence labels.

    Returns:
        Image with visualized detections.
    """
    output = image.copy()
    
    for det in detections:
        bbox = det.get("bbox", [])
        confidence = det.get("confidence", 0.0)
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_labels:
                label = f"Pool: {confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(
                    output, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
    
    return output
