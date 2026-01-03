"""
Postprocessing Module for Swimming Pool Detection System.

This module provides functions for coordinate extraction,
boundary drawing, and output file generation.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def extract_boundary_coordinates(
    bbox: np.ndarray,
    img_shape: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Convert bounding box to four corner points.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2].
        img_shape: Original image shape (height, width).

    Returns:
        List of (x, y) coordinates for corners.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Clamp to image boundaries
    height, width = img_shape
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Return corners: top-left, top-right, bottom-right, bottom-left
    coordinates = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x2, y2),  # bottom-right
        (x1, y2),  # bottom-left
    ]
    
    return coordinates


def draw_pool_outline(
    image: np.ndarray,
    coordinates: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
    thickness: int = 3
) -> np.ndarray:
    """
    Draw pool outline on image.

    Args:
        image: Input image as numpy array (BGR).
        coordinates: List of (x, y) boundary coordinates.
        color: BGR color for outline (default: blue).
        thickness: Line thickness in pixels.

    Returns:
        Image with drawn outline.
    """
    # Make a copy to avoid modifying original
    output = image.copy()
    
    # Convert coordinates to numpy array for cv2
    pts = np.array(coordinates, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw polygon
    cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
    
    return output


def draw_pool_outline_with_label(
    image: np.ndarray,
    coordinates: List[Tuple[int, int]],
    confidence: float,
    color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
    thickness: int = 3
) -> np.ndarray:
    """
    Draw pool outline with confidence label.

    Args:
        image: Input image as numpy array (BGR).
        coordinates: List of (x, y) boundary coordinates.
        confidence: Detection confidence score.
        color: BGR color for outline.
        thickness: Line thickness.

    Returns:
        Image with outline and label.
    """
    output = draw_pool_outline(image, coordinates, color, thickness)
    
    # Add label
    if coordinates:
        x, y = coordinates[0]  # Top-left corner
        label = f"Pool: {confidence:.2f}"
        
        # Label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            output,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color,
            -1
        )
        
        cv2.putText(
            output,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return output


def save_coordinates(
    coordinates: List[Tuple[int, int]],
    output_path: str,
    confidence: float
) -> None:
    """
    Save pool boundary coordinates to file.

    Args:
        coordinates: List of (x, y) boundary coordinates.
        output_path: Path to save coordinates.txt.
        confidence: Detection confidence score.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Pool Detection Coordinates\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write("Boundary Points:\n")
        
        for x, y in coordinates:
            f.write(f"{x},{y}\n")
    
    logger.info(f"Saved coordinates to: {output_path}")


def load_coordinates(file_path: str) -> Tuple[List[Tuple[int, int]], float]:
    """
    Load coordinates from file.

    Args:
        file_path: Path to coordinates.txt file.

    Returns:
        Tuple of (coordinates list, confidence).
    """
    coordinates = []
    confidence = 0.0
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Confidence:"):
            confidence = float(line.split(":")[1].strip())
        elif "," in line:
            x, y = line.split(",")
            coordinates.append((int(x), int(y)))
    
    return coordinates, confidence


def bbox_to_polygon(
    bbox: Union[List[float], np.ndarray]
) -> List[Tuple[int, int]]:
    """
    Convert bounding box to polygon coordinates.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2].

    Returns:
        List of polygon corner points.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    return [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ]


def scale_coordinates(
    coordinates: List[Tuple[int, int]],
    scale_x: float,
    scale_y: float
) -> List[Tuple[int, int]]:
    """
    Scale coordinates by given factors.

    Args:
        coordinates: List of (x, y) coordinates.
        scale_x: X-axis scaling factor.
        scale_y: Y-axis scaling factor.

    Returns:
        Scaled coordinates.
    """
    return [(int(x * scale_x), int(y * scale_y)) for x, y in coordinates]


def normalize_coordinates(
    coordinates: List[Tuple[int, int]],
    img_width: int,
    img_height: int
) -> List[Tuple[float, float]]:
    """
    Normalize coordinates to [0, 1] range.

    Args:
        coordinates: List of (x, y) pixel coordinates.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Normalized coordinates.
    """
    return [(x / img_width, y / img_height) for x, y in coordinates]


def denormalize_coordinates(
    coordinates: List[Tuple[float, float]],
    img_width: int,
    img_height: int
) -> List[Tuple[int, int]]:
    """
    Convert normalized coordinates to pixel coordinates.

    Args:
        coordinates: List of normalized (x, y) coordinates.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Pixel coordinates.
    """
    return [(int(x * img_width), int(y * img_height)) for x, y in coordinates]


def calculate_pool_area(coordinates: List[Tuple[int, int]]) -> float:
    """
    Calculate pool area from boundary coordinates using Shoelace formula.

    Args:
        coordinates: List of (x, y) coordinates forming a closed polygon.

    Returns:
        Area in square pixels.
    """
    n = len(coordinates)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[j][0] * coordinates[i][1]
    
    return abs(area) / 2.0


def calculate_pool_dimensions(
    coordinates: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Calculate approximate pool dimensions (width, height).

    Args:
        coordinates: List of (x, y) coordinates.

    Returns:
        Tuple of (width, height) in pixels.
    """
    if not coordinates:
        return (0, 0)
    
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    return (width, height)
