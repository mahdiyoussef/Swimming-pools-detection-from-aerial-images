"""
Postprocessing Module for Swimming Pool Detection System.
==========================================================

This module provides functions for processing detection results and
generating output files. It handles coordinate extraction, boundary
drawing, and saving coordinates to text files.

Key Functions:
    - extract_boundary_coordinates: Convert bbox [x1,y1,x2,y2] to corner points
    - draw_pool_outline: Draw polygon outline on image
    - save_coordinates: Save pool coordinates to text file
    - calculate_pool_area: Calculate pool area using Shoelace formula
    - calculate_pool_dimensions: Get pool width and height

Output Format:
    The coordinates.txt file contains:
    ```
    Pool Detection Coordinates
    Confidence: 0.8506
    Boundary Points:
    139,75
    175,75
    175,117
    139,117
    ```

Dependencies:
    - opencv-python: Image drawing operations
    - numpy: Array operations

Author: Swimming Pool Detection Team
Date: 2026-01-02
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging           # Logging functionality
import math              # Mathematical operations
import json              # JSON data handling
from pathlib import Path # Cross-platform file path handling
from typing import List, Tuple, Dict, Union  # Type hints

import cv2               # OpenCV for image drawing operations
import numpy as np       # NumPy for array operations

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging with timestamp and level information
logging.basicConfig(
    level=logging.INFO,                                    # Log INFO and above
    format="[%(asctime)s] [%(levelname)s] %(message)s",   # Log format
    datefmt="%Y-%m-%d %H:%M:%S"                           # Timestamp format
)
logger = logging.getLogger(__name__)  # Get logger for this module


# =============================================================================
# COORDINATE EXTRACTION FUNCTIONS
# =============================================================================

def extract_boundary_coordinates(
    bbox: np.ndarray,
    img_shape: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Convert bounding box to four corner point coordinates.
    
    This function takes a bounding box in [x1, y1, x2, y2] format and
    converts it to a list of four corner points, clipped to image boundaries.
    
    Coordinate System:
        (0,0) -------- X (width) --------->
          |
          |    (x1,y1) -------- (x2,y1)
          |       |                |
        Y |       |     BBOX       |
     (height)     |                |
          |    (x1,y2) -------- (x2,y2)
          |
          v
    
    Args:
        bbox (np.ndarray): Bounding box as [x1, y1, x2, y2] where:
            - x1, y1: Top-left corner coordinates (pixels)
            - x2, y2: Bottom-right corner coordinates (pixels)
        img_shape (Tuple[int, int]): Original image shape as (height, width).
            Used for clamping coordinates to valid range.
    
    Returns:
        List[Tuple[int, int]]: List of 4 corner coordinates in order:
            - (x1, y1): Top-left corner
            - (x2, y1): Top-right corner
            - (x2, y2): Bottom-right corner
            - (x1, y2): Bottom-left corner
    
    Example:
        >>> bbox = np.array([100, 50, 200, 150])
        >>> coords = extract_boundary_coordinates(bbox, (480, 640))
        >>> print(coords)
        [(100, 50), (200, 50), (200, 150), (100, 150)]
    """
    # Convert bbox values to integers (may be floats from model output)
    # Using map() applies int() to each of the first 4 values
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Extract image dimensions from shape tuple
    height, width = img_shape
    
    # Clamp coordinates to valid image boundaries
    # max(0, ...) ensures coordinate is not negative
    # min(..., dimension) ensures coordinate doesn't exceed image size
    x1 = max(0, min(x1, width))   # Clamp x1 to [0, width]
    y1 = max(0, min(y1, height))  # Clamp y1 to [0, height]
    x2 = max(0, min(x2, width))   # Clamp x2 to [0, width]
    y2 = max(0, min(y2, height))  # Clamp y2 to [0, height]
    
    # Create list of corner coordinates in clockwise order
    # This order is important for cv2.polylines to draw correctly
    coordinates = [
        (x1, y1),  # Top-left corner (anchor point)
        (x2, y1),  # Top-right corner
        (x2, y2),  # Bottom-right corner
        (x1, y2),  # Bottom-left corner
    ]
    
    return coordinates


def extract_pool_contour(
    image: np.ndarray,
    bbox: np.ndarray,
    epsilon_factor: float = 0.012,
    min_area_ratio: float = 0.05  # Lowered to be more inclusive
) -> List[Tuple[int, int]]:
    """
    Extract high-precision pool contour using GrabCut and color refinement.
    
    This function combines GrabCut segmentation with HSV color priors to extract
    the actual pool boundary within the detection bounding box.
    
    Args:
        image (np.ndarray): Original image in BGR format.
        bbox (np.ndarray): Detection box [x1, y1, x2, y2].
        epsilon_factor (float): Simplification tolerance.
        min_area_ratio (float): Minimum area check.
    
    Returns:
        List[Tuple[int, int]]: High-precision polygon vertices.
    """
    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox[:4])
    h_img, w_img = image.shape[:2]
    
    # Increase padding to 15 pixels to capture full pool boundaries
    pad = 15
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    
    crop = image[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    # 1. HSV Color Mask for Priors
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Refined water color ranges (including pool covers and shadows)
    lower_water = np.array([75, 20, 40])
    upper_water = np.array([150, 255, 255])
    color_mask = cv2.inRange(hsv, lower_water, upper_water)
    
    # 2. GrabCut Initialization
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Rect relative to crop: the original bbox area
    rx, ry = max(0, x1 - cx1), max(0, y1 - cy1)
    rw, rh = x2 - x1, y2 - y1
    rect = (rx, ry, rw, rh)
    
    # Initial pass with rect
    try:
        cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
        
        # Refine with color priors
        mask[(mask == cv2.GC_PR_FGD) & (color_mask > 0)] = cv2.GC_FGD
        mask[(mask == cv2.GC_PR_BGD) & (color_mask > 0)] = cv2.GC_PR_FGD
        
        # Second pass with mask
        cv2.grabCut(crop, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    except:
        # Fallback to pure color mask if GrabCut fails
        bin_mask = color_mask
    
    # 3. Handle Edge cases - if GrabCut/Color results in empty mask, use adaptive threshold
    if np.sum(bin_mask) < (rw * rh * 255 * min_area_ratio):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        bin_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        # Apply mask to keep only what's inside the padded bbox
        box_mask = np.zeros_like(bin_mask)
        cv2.rectangle(box_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)
        bin_mask = cv2.bitwise_and(bin_mask, box_mask)
    
    # 4. Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. Extract Contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    if not contours:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
    # Select largest contour that overlaps with our detector's bbox
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (rw * rh * min_area_ratio):
            valid_contours.append(cnt)
            
    if not valid_contours:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
    cnt = max(valid_contours, key=cv2.contourArea)
    
    # 6. Smooth and Simplify
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
    
    # Convert to global coordinates
    pts = []
    for p in approx:
        gx = max(0, min(w_img - 1, int(p[0][0]) + cx1))
        gy = max(0, min(h_img - 1, int(p[0][1]) + cy1))
        pts.append((gx, gy))
        
    if len(pts) < 3:
         return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
         
    return pts
    
    # If no contours found, return bbox corners
    if not contours:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    # Select largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough
    bbox_area = (x2 - x1) * (y2 - y1)
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < bbox_area * min_area_ratio:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    # Simplify contour using Douglas-Peucker algorithm
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = epsilon_factor * perimeter
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of tuples and offset to original image coordinates
    contour_points = []
    for point in simplified:
        px = int(point[0][0]) + crop_x1
        py = int(point[0][1]) + crop_y1
        # Clamp to image bounds
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        contour_points.append((px, py))
    
    # Ensure we have at least 3 points
    if len(contour_points) < 3:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    logger.debug(f"Extracted contour with {len(contour_points)} points")
    return contour_points


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_pool_outline(
    image: np.ndarray,
    coordinates: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR format
    thickness: int = 3
) -> np.ndarray:
    """
    Draw pool outline on image using polygon lines.
    
    This function draws a closed polygon outline around the detected pool
    using OpenCV's polylines function. The outline connects all corner
    points in order and closes back to the starting point.
    
    Args:
        image (np.ndarray): Input image as numpy array in BGR format.
            Shape: (height, width, 3) for color images.
        coordinates (List[Tuple[int, int]]): List of (x, y) boundary coordinates.
            Expected to have 4 points for rectangular pools.
        color (Tuple[int, int, int], optional): BGR color for the outline.
            OpenCV uses BGR format, not RGB!
            Common colors:
            - (0, 0, 255): Red
            - (0, 255, 0): Green
            - (255, 0, 0): Blue
            - (0, 255, 255): Yellow
            Defaults to (0, 0, 255) - Red.
        thickness (int, optional): Line thickness in pixels.
            Thicker lines are more visible but may obscure pool edges.
            Defaults to 3.
    
    Returns:
        np.ndarray: Copy of image with drawn outline.
            Original image is not modified.
    
    Example:
        >>> coords = [(100, 50), (200, 50), (200, 150), (100, 150)]
        >>> output = draw_pool_outline(image, coords, color=(0, 0, 255))
    """
    # Create a copy to avoid modifying the original image
    # This is important for non-destructive drawing
    output = image.copy()
    
    # Convert Python list of tuples to numpy array for OpenCV
    # dtype=np.int32 is required by cv2.polylines
    pts = np.array(coordinates, dtype=np.int32)
    
    # Reshape to format required by cv2.polylines: (N, 1, 2)
    # N = number of points, 1 = contour count, 2 = x,y coordinates
    pts = pts.reshape((-1, 1, 2))
    
    # Draw the polygon outline
    # - [pts]: List of polygons (we have one)
    # - isClosed=True: Connect last point to first point
    # - color: BGR color tuple
    # - thickness: Line width in pixels
    cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
    
    return output


def draw_pool_outline_with_label(
    image: np.ndarray,
    coordinates: List[Tuple[int, int]],
    confidence: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw pool outline with confidence label overlay.
    
    This function extends draw_pool_outline by adding a text label
    showing the detection confidence score. The label has a colored
    background for visibility against any background.
    
    Label Format: "Pool: 0.95" (confidence rounded to 2 decimals)
    
    Args:
        image (np.ndarray): Input image as numpy array (BGR format).
        coordinates (List[Tuple[int, int]]): List of (x, y) boundary coordinates.
        confidence (float): Detection confidence score (0.0 to 1.0).
            Displayed as "Pool: X.XX" above the bounding box.
        color (Tuple[int, int, int], optional): BGR color for outline and label.
            Defaults to (0, 0, 255) - Red.
        thickness (int, optional): Line thickness. Defaults to 3.
    
    Returns:
        np.ndarray: Image with outline and label drawn.
    
    Example:
        >>> output = draw_pool_outline_with_label(
        ...     image, coords, confidence=0.92, color=(0, 255, 0)
        ... )
    """
    # First, draw the basic outline
    output = draw_pool_outline(image, coordinates, color, thickness)
    
    # Add confidence label if coordinates exist
    if coordinates:
        # Position label at top-left corner of bounding box
        x, y = coordinates[0]  # Top-left corner
        
        # Format label text
        label = f"Pool: {confidence:.2f}"
        
        # Calculate text size for background rectangle
        # getTextSize returns ((width, height), baseline)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,                         # Text string
            cv2.FONT_HERSHEY_SIMPLEX,     # Font type
            0.6,                           # Font scale
            2                              # Thickness
        )
        
        # Draw filled rectangle as label background
        # This makes text readable on any background
        cv2.rectangle(
            output,
            (x, y - text_height - 10),         # Top-left of rectangle
            (x + text_width + 10, y),          # Bottom-right of rectangle
            color,                              # Rectangle color
            -1                                  # -1 means filled rectangle
        )
        
        # Draw white text on colored background
        cv2.putText(
            output,
            label,                              # Text string
            (x + 5, y - 5),                    # Position (slightly inside rect)
            cv2.FONT_HERSHEY_SIMPLEX,          # Font type
            0.6,                                # Font scale
            (255, 255, 255),                   # White text color
            2                                   # Text thickness
        )
    
    return output


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def save_coordinates(
    coordinates: List[Tuple[int, int]],
    output_path: str,
    confidence: float
) -> None:
    """
    Save pool boundary coordinates to a text file.
    
    Creates a human-readable text file containing the pool detection
    information including confidence score and boundary point coordinates.
    
    Output File Format:
        Line 1: "Pool Detection Coordinates"
        Line 2: "Confidence: X.XXXX"
        Line 3: "Boundary Points:"
        Lines 4+: "X,Y" for each corner point
    
    Args:
        coordinates (List[Tuple[int, int]]): List of (x, y) boundary coordinates.
            Each tuple represents a corner point of the pool boundary.
        output_path (str): Path to save the coordinates.txt file.
            Parent directories are created if they don't exist.
        confidence (float): Detection confidence score (0.0 to 1.0).
            Saved with 4 decimal precision.
    
    Example:
        >>> coords = [(100, 50), (200, 50), (200, 150), (100, 150)]
        >>> save_coordinates(coords, "output/coordinates.txt", 0.95)
        # Creates file with:
        # Pool Detection Coordinates
        # Confidence: 0.9500
        # Boundary Points:
        # 100,50
        # 200,50
        # 200,150
        # 100,150
    """
    # Convert to Path object for robust path handling
    output_path = Path(output_path)
    
    # Create parent directories if they don't exist
    # parents=True creates all intermediate directories
    # exist_ok=True doesn't raise error if directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write coordinates to file
    with open(output_path, "w") as f:
        # Write header
        f.write("Pool Detection Coordinates\n")
        
        # Write confidence with 4 decimal places
        f.write(f"Confidence: {confidence:.4f}\n")
        
        # Write boundary points section
        f.write("Boundary Points:\n")
        
        # Write each coordinate pair on its own line
        for x, y in coordinates:
            f.write(f"{x},{y}\n")
    
    # Log the save operation
    logger.info(f"Saved coordinates to: {output_path}")


def load_coordinates(file_path: str) -> Tuple[List[Tuple[int, int]], float]:
    """
    Load coordinates from a previously saved coordinates.txt file.
    
    Parses the coordinate file format and extracts the boundary points
    and confidence score. Useful for re-processing or visualization.
    
    Args:
        file_path (str): Path to coordinates.txt file.
    
    Returns:
        Tuple[List[Tuple[int, int]], float]: A tuple containing:
            - List of (x, y) coordinate tuples
            - Confidence score as float
    
    Example:
        >>> coords, conf = load_coordinates("output/coordinates.txt")
        >>> print(f"Loaded {len(coords)} points with confidence {conf}")
    """
    coordinates = []     # List to store parsed coordinates
    confidence = 0.0     # Default confidence
    
    # Read all lines from file
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Parse each line
    for line in lines:
        line = line.strip()  # Remove whitespace and newlines
        
        # Parse confidence line
        if line.startswith("Confidence:"):
            # Extract number after colon
            confidence = float(line.split(":")[1].strip())
        
        # Parse coordinate lines (contain comma)
        elif "," in line:
            # Split by comma and convert to integers
            x, y = line.split(",")
            coordinates.append((int(x), int(y)))
    
    return coordinates, confidence


# =============================================================================
# COORDINATE CONVERSION FUNCTIONS
# =============================================================================

def bbox_to_polygon(
    bbox: Union[List[float], np.ndarray]
) -> List[Tuple[int, int]]:
    """
    Convert bounding box to polygon corner coordinates.
    
    Simple conversion without image boundary clamping. Use
    extract_boundary_coordinates() if you need boundary checking.
    
    Args:
        bbox (Union[List[float], np.ndarray]): Bounding box as [x1, y1, x2, y2].
    
    Returns:
        List[Tuple[int, int]]: List of 4 corner points:
            [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    
    Example:
        >>> polygon = bbox_to_polygon([100.5, 50.5, 200.5, 150.5])
        >>> print(polygon)
        [(100, 50), (200, 50), (200, 150), (100, 150)]
    """
    # Convert to integers (truncates decimals)
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Return corners in clockwise order
    return [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x2, y2),  # Bottom-right
        (x1, y2)   # Bottom-left
    ]


def scale_coordinates(
    coordinates: List[Tuple[int, int]],
    scale_x: float,
    scale_y: float
) -> List[Tuple[int, int]]:
    """
    Scale coordinates by given factors.
    
    Useful for converting coordinates between different image resolutions.
    For example, if detection was done on a resized image, scale
    coordinates to match the original image size.
    
    Formula: new_coord = old_coord * scale_factor
    
    Args:
        coordinates (List[Tuple[int, int]]): Original (x, y) coordinates.
        scale_x (float): X-axis scaling factor.
            > 1.0 enlarges, < 1.0 shrinks.
        scale_y (float): Y-axis scaling factor.
            > 1.0 enlarges, < 1.0 shrinks.
    
    Returns:
        List[Tuple[int, int]]: Scaled coordinates.
    
    Example:
        >>> # Scale from 320x240 to 640x480
        >>> scaled = scale_coordinates(coords, 2.0, 2.0)
    """
    # Apply scale factors and convert to integers
    return [(int(x * scale_x), int(y * scale_y)) for x, y in coordinates]


def normalize_coordinates(
    coordinates: List[Tuple[int, int]],
    img_width: int,
    img_height: int
) -> List[Tuple[float, float]]:
    """
    Normalize coordinates to [0, 1] range.
    
    Converts pixel coordinates to normalized coordinates where:
    - (0, 0) is the top-left corner
    - (1, 1) is the bottom-right corner
    
    This is useful for format conversion (e.g., to YOLO format).
    
    Formula:
        normalized_x = pixel_x / image_width
        normalized_y = pixel_y / image_height
    
    Args:
        coordinates (List[Tuple[int, int]]): List of (x, y) pixel coordinates.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.
    
    Returns:
        List[Tuple[float, float]]: Normalized coordinates in [0, 1] range.
    
    Example:
        >>> # 640x480 image, point at center
        >>> norm = normalize_coordinates([(320, 240)], 640, 480)
        >>> print(norm)
        [(0.5, 0.5)]
    """
    return [(x / img_width, y / img_height) for x, y in coordinates]


def denormalize_coordinates(
    coordinates: List[Tuple[float, float]],
    img_width: int,
    img_height: int
) -> List[Tuple[int, int]]:
    """
    Convert normalized [0, 1] coordinates back to pixel coordinates.
    
    Inverse of normalize_coordinates(). Converts normalized coordinates
    to pixel coordinates for a specific image size.
    
    Formula:
        pixel_x = normalized_x * image_width
        pixel_y = normalized_y * image_height
    
    Args:
        coordinates (List[Tuple[float, float]]): Normalized (x, y) coordinates.
        img_width (int): Target image width in pixels.
        img_height (int): Target image height in pixels.
    
    Returns:
        List[Tuple[int, int]]: Pixel coordinates.
    
    Example:
        >>> # Convert normalized center to 640x480 pixels
        >>> pixels = denormalize_coordinates([(0.5, 0.5)], 640, 480)
        >>> print(pixels)
        [(320, 240)]
    """
    return [(int(x * img_width), int(y * img_height)) for x, y in coordinates]


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

def calculate_pool_area(coordinates: List[Tuple[int, int]]) -> float:
    """
    Calculate pool area from boundary coordinates using Shoelace formula.
    
    The Shoelace formula (also known as Gauss's area formula) calculates
    the area of a simple polygon given its vertices. It works for any
    non-self-intersecting polygon, not just rectangles.
    
    Formula (Shoelace):
        Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    
    Args:
        coordinates (List[Tuple[int, int]]): List of (x, y) coordinates
            forming a closed polygon. Minimum 3 points required.
    
    Returns:
        float: Area in square pixels.
            Returns 0.0 if less than 3 coordinates provided.
    
    Note:
        To convert to real-world area (e.g., square meters), multiply
        by (ground_resolution)^2 where ground_resolution is meters/pixel.
    
    Example:
        >>> # 100x100 pixel rectangle = 10000 sq pixels
        >>> coords = [(0, 0), (100, 0), (100, 100), (0, 100)]
        >>> area = calculate_pool_area(coords)
        >>> print(area)
        10000.0
    """
    n = len(coordinates)
    
    # Need at least 3 points to form a polygon
    if n < 3:
        return 0.0
    
    # Apply Shoelace formula
    area = 0.0
    for i in range(n):
        # Next index (wraps around to 0 for last point)
        j = (i + 1) % n
        
        # Shoelace: sum of (x_i * y_{i+1}) - (x_{i+1} * y_i)
        area += coordinates[i][0] * coordinates[j][1]  # x_i * y_{i+1}
        area -= coordinates[j][0] * coordinates[i][1]  # x_{i+1} * y_i
    
    # Take absolute value and divide by 2
    return abs(area) / 2.0


# =============================================================================
# GEOSPATIAL FUNCTIONS
# =============================================================================

def pixel_to_latlng(
    pixel_coords: List[Tuple[int, int]],
    origin_lat: float,
    origin_lng: float,
    meters_per_pixel: float
) -> List[Tuple[float, float]]:
    """
    Convert pixel coordinates to real-world Latitude and Longitude.
    
    Uses a planar approximation suitable for small-scale aerial imagery.
    Accounts for the convergence of meridians (cosine of latitude).
    
    Args:
        pixel_coords: List of (x, y) coordinates in pixels.
        origin_lat: Latitude of the top-left corner (pixel 0,0).
        origin_lng: Longitude of the top-left corner (pixel 0,0).
        meters_per_pixel: Ground resolution (e.g., 0.3 for 30cm/px).
        
    Returns:
        List of (latitude, longitude) tuples.
    """
    # Earth's radius constants in meters
    DEG_TO_METERS = 111320.0
    
    # Calculate scale factors
    # Latitude degrees are constant: 1 deg ~ 111.32 km
    lat_scale = meters_per_pixel / DEG_TO_METERS
    
    # Longitude degrees vary with latitude: 1 deg ~ 111.32 * cos(lat) km
    lng_scale = meters_per_pixel / (DEG_TO_METERS * math.cos(math.radians(origin_lat)))
    
    geo_coords = []
    for x, y in pixel_coords:
        # Note: In screen space, +Y is South (decreasing Lat)
        # and +X is East (increasing Lng)
        lat = origin_lat - (y * lat_scale)
        lng = origin_lng + (x * lng_scale)
        geo_coords.append((lat, lng))
        
    return geo_coords


def calculate_real_world_area(
    pixel_area: float,
    meters_per_pixel: float
) -> float:
    """
    Convert pixel area to square meters.
    
    Args:
        pixel_area: Area in square pixels.
        meters_per_pixel: Ground resolution in meters.
        
    Returns:
        Area in square meters.
    """
    return pixel_area * (meters_per_pixel ** 2)


def save_as_geojson(
    detections: List[Dict],
    output_path: str,
    origin_lat: float,
    origin_lng: float,
    meters_per_pixel: float
) -> None:
    """
    Save detections as a GeoJSON FeatureCollection for map visualization.
    
    Args:
        detections: List of detection dictionaries containing "polygon", "confidence", "shape".
        output_path: Destination file path.
        origin_lat: Image origin latitude.
        origin_lng: Image origin longitude.
        meters_per_pixel: Ground resolution.
    """
    features = []
    
    for i, det in enumerate(detections):
        # Convert polygon pixels to Lat/Lng
        geo_poly = pixel_to_latlng(det["polygon"], origin_lat, origin_lng, meters_per_pixel)
        
        # Calculate real area
        pix_area = calculate_pool_area(det["polygon"])
        real_area = calculate_real_world_area(pix_area, meters_per_pixel)
        
        # GeoJSON expects [Longitude, Latitude] and closed loop
        coordinates = [[lng, lat] for lat, lng in geo_poly]
        if coordinates:
            coordinates.append(coordinates[0]) # Close the loop
            
        feature = {
            "type": "Feature",
            "id": i,
            "properties": {
                "confidence": float(det["confidence"]),
                "shape": det["shape"],
                "area_m2": round(real_area, 2),
                "vertex_count": len(det["polygon"])
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
        }
        features.append(feature)
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def calculate_pool_dimensions(
    coordinates: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Calculate approximate pool dimensions (width, height).
    
    Computes the bounding box dimensions of the polygon formed by
    the coordinates. This gives the maximum width and height.
    
    Args:
        coordinates (List[Tuple[int, int]]): List of (x, y) coordinates.
    
    Returns:
        Tuple[int, int]: (width, height) in pixels.
            Returns (0, 0) if no coordinates provided.
    
    Note:
        For rotated pools, this returns the bounding box dimensions,
        not the actual pool dimensions along its major/minor axes.
    
    Example:
        >>> coords = [(100, 50), (200, 50), (200, 150), (100, 150)]
        >>> w, h = calculate_pool_dimensions(coords)
        >>> print(f"Width: {w}, Height: {h}")
        Width: 100, Height: 100
    """
    # Handle empty input
    if not coordinates:
        return (0, 0)
    
    # Extract x and y coordinates separately
    x_coords = [c[0] for c in coordinates]  # All x values
    y_coords = [c[1] for c in coordinates]  # All y values
    
    # Calculate dimensions from min/max values
    width = max(x_coords) - min(x_coords)    # Horizontal span
    height = max(y_coords) - min(y_coords)   # Vertical span
    
    return (width, height)


# =============================================================================
# SEGMENTATION-SPECIFIC FUNCTIONS
# =============================================================================

def draw_pool_mask(
    image: np.ndarray,
    polygon: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
    alpha: float = 0.4,
    outline_thickness: int = 2
) -> np.ndarray:
    """
    Draw a filled semi-transparent mask for a segmented pool.
    
    This function draws both a filled polygon mask and an outline,
    creating a visually appealing segmentation visualization.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        polygon (List[Tuple[float, float]]): List of (x, y) polygon vertices.
            Can have variable number of vertices for different pool shapes.
        color (Tuple[int, int, int], optional): BGR color for the mask.
            Defaults to (0, 255, 0) - Green.
        alpha (float, optional): Transparency of the filled mask.
            0.0 = fully transparent, 1.0 = fully opaque.
            Defaults to 0.4.
        outline_thickness (int, optional): Thickness of outline in pixels.
            Defaults to 2.
    
    Returns:
        np.ndarray: Image with mask overlay drawn.
    
    Example:
        >>> polygon = [[100, 50], [150, 40], [200, 55], [190, 150], [100, 140]]
        >>> output = draw_pool_mask(image, polygon, color=(0, 255, 0), alpha=0.4)
    """
    # Create a copy to avoid modifying the original
    output = image.copy()
    overlay = image.copy()
    
    # Convert polygon to numpy array of integers
    pts = np.array(polygon, dtype=np.int32)
    
    # Draw filled polygon on overlay
    cv2.fillPoly(overlay, [pts], color)
    
    # Blend overlay with original image using alpha
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # Draw outline on top
    cv2.polylines(output, [pts], isClosed=True, color=color, thickness=outline_thickness)
    
    return output


def calculate_circularity(polygon: List[Tuple[float, float]]) -> float:
    """
    Calculate the circularity of a polygon shape.
    
    Circularity is defined as: 4 * pi * area / perimeter^2
    A perfect circle has circularity = 1.0
    A square has circularity ~ 0.785
    More irregular shapes have lower values.
    
    Args:
        polygon (List[Tuple[float, float]]): List of (x, y) polygon vertices.
    
    Returns:
        float: Circularity value between 0.0 and 1.0.
            Returns 0.0 for invalid polygons.
    
    Example:
        >>> # Roughly circular polygon
        >>> circ = calculate_circularity(circular_polygon)
        >>> print(f"Circularity: {circ:.2f}")
        Circularity: 0.92
    """
    if len(polygon) < 3:
        return 0.0
    
    # Convert to numpy array
    pts = np.array(polygon, dtype=np.float32)
    
    # Calculate area using Shoelace formula
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    area = abs(area) / 2.0
    
    # Calculate perimeter
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        perimeter += np.sqrt(dx * dx + dy * dy)
    
    # Calculate circularity
    if perimeter == 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    
    # Clamp to valid range
    return min(1.0, max(0.0, circularity))


def classify_pool_shape(
    polygon: List[Tuple[float, float]],
    circularity_threshold: float = 0.85,
    rectangular_threshold: float = 0.90
) -> str:
    """
    Classify the shape of a pool based on its polygon contour.
    
    Shape categories:
        - "rectangular": 4 vertices, high rectangularity
        - "circular": High circularity (>0.85)
        - "oval": Medium-high circularity (0.70-0.85)
        - "irregular": Complex shapes
    
    Args:
        polygon (List[Tuple[float, float]]): List of (x, y) polygon vertices.
        circularity_threshold (float): Threshold for classifying as circular.
            Defaults to 0.85.
        rectangular_threshold (float): Aspect ratio threshold for rectangles.
            Defaults to 0.90.
    
    Returns:
        str: Shape classification: "rectangular", "circular", "oval", or "irregular"
    
    Example:
        >>> shape = classify_pool_shape(pool_polygon)
        >>> print(f"Pool shape: {shape}")
        Pool shape: oval
    """
    if len(polygon) < 3:
        return "irregular"
    
    # Calculate circularity
    circularity = calculate_circularity(polygon)
    
    # Convert to numpy for analysis
    pts = np.array(polygon, dtype=np.float32)
    
    # Check for rectangular shape (4 vertices with right angles)
    if len(polygon) <= 5:
        # Check if it's approximately rectangular
        # Get bounding box and compare areas
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        bbox_area = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
        
        # Calculate polygon area
        polygon_area = calculate_pool_area([(p[0], p[1]) for p in polygon])
        
        # If polygon fills most of its bounding box, it's rectangular
        if bbox_area > 0 and polygon_area / bbox_area > rectangular_threshold:
            return "rectangular"
    
    # Check for circular shape
    if circularity >= circularity_threshold:
        return "circular"
    
    # Check for oval shape (moderate circularity)
    if circularity >= 0.70:
        return "oval"
    
    # Default to irregular for complex shapes
    return "irregular"


def simplify_polygon(
    polygon: List[Tuple[float, float]],
    epsilon_factor: float = 0.02
) -> List[Tuple[float, float]]:
    """
    Simplify a polygon using Douglas-Peucker algorithm.
    
    This is useful for reducing the number of vertices in complex
    polygon contours while preserving the overall shape.
    
    Args:
        polygon (List[Tuple[float, float]]): List of (x, y) polygon vertices.
        epsilon_factor (float): Approximation accuracy as a fraction of perimeter.
            Lower values = more detailed (more vertices).
            Higher values = more simplified (fewer vertices).
            Defaults to 0.02 (2% of perimeter).
    
    Returns:
        List[Tuple[float, float]]: Simplified polygon coordinates.
    
    Example:
        >>> simplified = simplify_polygon(complex_polygon, epsilon_factor=0.03)
        >>> print(f"Reduced from {len(complex_polygon)} to {len(simplified)} vertices")
    """
    if len(polygon) < 4:
        return polygon
    
    # Convert to numpy array
    pts = np.array(polygon, dtype=np.float32)
    
    # Calculate epsilon based on perimeter
    perimeter = cv2.arcLength(pts, closed=True)
    epsilon = epsilon_factor * perimeter
    
    # Apply Douglas-Peucker simplification
    simplified = cv2.approxPolyDP(pts, epsilon, closed=True)
    
    # Convert back to list of tuples
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def save_segmentation_coordinates(
    polygon: List[Tuple[float, float]],
    output_path: str,
    confidence: float,
    shape: str = "unknown"
) -> None:
    """
    Save pool segmentation coordinates to a text file.
    
    Extended version of save_coordinates for segmentation output.
    Includes shape classification and variable vertex count.
    
    Output File Format:
        Line 1: "Pool Segmentation Coordinates"
        Line 2: "Confidence: X.XXXX"
        Line 3: "Shape: <shape_type>"
        Line 4: "Vertex Count: N"
        Line 5: "Boundary Points:"
        Lines 6+: "X,Y" for each vertex
    
    Args:
        polygon (List[Tuple[float, float]]): List of (x, y) polygon vertices.
        output_path (str): Path to save the coordinates file.
        confidence (float): Detection confidence score (0.0 to 1.0).
        shape (str): Shape classification (rectangular, circular, oval, irregular).
    
    Example:
        >>> save_segmentation_coordinates(polygon, "output/coordinates.txt", 0.95, "oval")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Pool Segmentation Coordinates\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"Shape: {shape}\n")
        f.write(f"Vertex Count: {len(polygon)}\n")
        f.write("Boundary Points:\n")
        
        for point in polygon:
            x, y = int(point[0]), int(point[1])
            f.write(f"{x},{y}\n")
    
    logger.info(f"Saved segmentation coordinates to: {output_path}")

