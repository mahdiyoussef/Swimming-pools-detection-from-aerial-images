"""
Pool Detection Script for Swimming Pool Detection System.
==========================================================

This module provides the main inference pipeline for detecting swimming pools
in aerial/satellite imagery using trained YOLO26 deep learning models.

Features:
    - Single image detection
    - Batch processing of image directories  
    - Tiled/sliding window inference for large images (10000x10000+ pixels)
    - Non-Maximum Suppression for merging overlapping detections
    - Output generation: coordinates.txt and output_image.jpg

Output Files:
    - coordinates.txt: Pool boundary coordinates in pixel format
    - output_image.jpg: Original image with red bounding box overlays

Dependencies:
    - ultralytics: YOLO26 model inference
    - opencv-python: Image loading and processing
    - numpy: Array operations

Usage:
    # Standard detection
    python detect_pools.py --input image.jpg
    
    # Tiled detection for large images
    python detect_pools.py --input large_image.png --tiled
    
    # Custom confidence threshold
    python detect_pools.py --input image.jpg --conf-threshold 0.7

Author: Youssef Mahdi
Date: 2026-01-02
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse          # Command-line argument parsing
import logging           # Logging functionality
import sys               # System path manipulation
import time              # Timing inference operations
import shutil            # File and directory operations
from pathlib import Path # Cross-platform file path handling
from typing import Any, Dict, List, Optional, Tuple  # Type hints

# Add project root to Python path for local module imports
# This allows importing from inference/, preprocessing/, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2               # OpenCV for image I/O and drawing
import numpy as np       # NumPy for array operations

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging format with timestamp, level, and message
# Example output: [2026-01-20 15:30:45] [INFO] Loading model...
logging.basicConfig(
    level=logging.INFO,                                    # Log INFO and above
    format="[%(asctime)s] [%(levelname)s] %(message)s",   # Log message format
    datefmt="%Y-%m-%d %H:%M:%S"                           # Timestamp format
)
logger = logging.getLogger(__name__)  # Get logger for this module


# =============================================================================
# POOL DETECTOR CLASS
# =============================================================================

class PoolDetector:
    """
    Swimming Pool Detector using YOLO26.
    
    This class encapsulates the YOLO26 model and provides methods for
    detecting swimming pools in images of various sizes. It supports
    standard detection for small images and tiled detection for large
    aerial/satellite imagery.
    
    Attributes:
        model (YOLO): Loaded YOLO26 model instance from ultralytics.
        model_path (Path): Path to the trained model weights (.pt file).
        conf_threshold (float): Minimum confidence score for valid detections.
            Detections below this threshold are filtered out.
            Range: 0.0 to 1.0, Default: 0.25
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.
            Higher values allow more overlapping boxes.
            Range: 0.0 to 1.0, Default: 0.45
        device (str): Computing device for inference.
            Options: "0", "1", etc. for GPU, "cpu" for CPU inference.
    
    Example:
        >>> detector = PoolDetector("weights/best.pt", conf_threshold=0.5)
        >>> image, detections = detector.detect("aerial_photo.jpg")
        >>> print(f"Found {len(detections)} swimming pools")
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "0"
    ) -> None:
        """
        Initialize the PoolDetector with model and configuration.
        
        Args:
            model_path (str): Path to YOLO26 weights file (.pt format).
                Must be a valid path to an existing file.
            conf_threshold (float, optional): Minimum confidence for detections.
                Lower values = more detections (but more false positives).
                Higher values = fewer detections (but higher precision).
                Defaults to 0.25.
            iou_threshold (float, optional): IoU threshold for NMS.
                Controls overlap allowed between detected boxes.
                Defaults to 0.45.
            device (str, optional): CUDA device ID or 'cpu'.
                Examples: "0" (first GPU), "1" (second GPU), "cpu".
                Defaults to "0".
        
        Raises:
            FileNotFoundError: If model_path does not exist.
        
        Example:
            >>> detector = PoolDetector(
            ...     model_path="weights/best.pt",
            ...     conf_threshold=0.5,
            ...     device="0"
            ... )
        """
        # Store configuration as instance attributes
        self.model_path = Path(model_path)      # Convert to Path object
        self.conf_threshold = conf_threshold     # Store confidence threshold
        self.iou_threshold = iou_threshold       # Store IoU threshold
        self.device = device                     # Store device setting
        
        # Validate that model file exists before attempting to load
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load the YOLO26 model (lazy import to speed up module loading)
        self.model = self._load_model()

    def _load_model(self) -> "YOLO":
        """
        Load YOLO26 model from disk (private method).
        
        This method uses lazy importing of ultralytics to avoid
        loading the heavy library until actually needed.
        
        Returns:
            YOLO: Loaded YOLO26 model instance ready for inference.
        
        Note:
            Called automatically by __init__, no need to call directly.
        """
        # Import ultralytics here (lazy import) to speed up initial module load
        from ultralytics import YOLO
        
        # Log the model loading for debugging
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load the model weights
        model = YOLO(str(self.model_path))
        
        return model

    def detect(
        self,
        image_input,
        image_size: int = 640
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run detection on a single image (standard size).
        
        This method is optimized for images close to the training size
        (e.g., 512x512 or 640x640). For larger images (1000+ pixels),
        use detect_tiled() instead for better accuracy.
        
        Args:
            image_input: Either a path to an image file (str/Path) OR
                a numpy array (BGR format) representing the image directly.
                Supported formats for paths: .jpg, .jpeg, .png, .bmp
            image_size (int, optional): Size to resize image for inference.
                Larger values = more detail but slower inference.
                Common values: 416, 512, 640, 1280.
                Defaults to 640.
        
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: A tuple containing:
                - np.ndarray: Original image in BGR format (OpenCV standard)
                - List[Dict]: List of detection dictionaries, each containing:
                    - "bbox": [x1, y1, x2, y2] bounding box in pixels
                    - "confidence": float (0.0 to 1.0) detection confidence
                    - "class_id": int (0 for swimming_pool)
                    - "class_name": str ("swimming_pool")
        
        Raises:
            FileNotFoundError: If image_input is a path that does not exist.
            ValueError: If image cannot be loaded (corrupted/unsupported format).
        
        Example:
            >>> image, detections = detector.detect("test.jpg", image_size=640)
            >>> for det in detections:
            ...     print(f"Pool at {det['bbox']} with conf {det['confidence']:.2f}")
        """
        # -------------------------------------------------------------------
        # STEP 1: Load the original image (from path or use array directly)
        # -------------------------------------------------------------------
        if isinstance(image_input, np.ndarray):
            # Input is already a numpy array (e.g., from fetch_map_frame)
            original_image = image_input
        else:
            # Input is a file path
            image_path = Path(image_input)
            
            # Validate image exists
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # cv2.imread returns BGR format (Blue, Green, Red)
            original_image = cv2.imread(str(image_path))
            
            # Check if image was loaded successfully
            if original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        
        # -------------------------------------------------------------------
        # STEP 2: Run YOLO26 inference
        # -------------------------------------------------------------------
        start_time = time.time()  # Start timing
        
        # Run model prediction
        # - source: input image (can be path or numpy array)
        # - conf: confidence threshold (filter low-confidence detections)
        # - iou: IoU threshold for NMS (filter overlapping boxes)
        # - imgsz: resize image to this size for inference
        # - device: GPU device or CPU
        # - verbose: suppress ultralytics output
        results = self.model.predict(
            source=original_image,  # Pass the numpy array directly
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False
        )
        
        # Calculate and log inference time
        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time * 1000:.2f} ms")
        
        # -------------------------------------------------------------------
        # STEP 3: Process detection/segmentation results
        # -------------------------------------------------------------------
        detections = []  # List to store all detections
        
        # Iterate through results (usually just one result for single image)
        for result in results:
            boxes = result.boxes  # Get Boxes object containing all detections
            masks = result.masks  # Get Masks object (None for detection-only models)
            
            # Skip if no detections
            if boxes is None:
                continue
            
            # Check if this is a segmentation result (has masks)
            has_masks = masks is not None and len(masks) > 0
            
            # Process each detected box
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract bounding box coordinates in xyxy format
                # xyxy format: [x1, y1, x2, y2] where (x1,y1) is top-left
                # and (x2,y2) is bottom-right corner
                xyxy = box.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy
                
                # Extract confidence score (0.0 to 1.0)
                confidence = float(box.conf[0].cpu().numpy())
                
                # Extract class ID (0 = swimming_pool for this model)
                class_id = int(box.cls[0].cpu().numpy())
                
                # Create detection dictionary
                detection = {
                    "bbox": xyxy.tolist(),       # Bounding box [x1,y1,x2,y2]
                    "confidence": confidence,     # Detection confidence score
                    "class_id": class_id,         # Class index (0 = pool)
                    "class_name": "swimming_pool" # Human-readable class name
                }
                
                # -----------------------------------------------------------
                # Extract polygon mask (segmentation models only)
                # -----------------------------------------------------------
                if has_masks:
                    # Get polygon coordinates for this mask
                    # masks.xy contains list of polygon arrays (N,2) for each detection
                    polygon_xy = masks.xy[i]  # Pixel coordinates (N, 2)
                    
                    # Convert to list of (x, y) tuples
                    if len(polygon_xy) > 0:
                        detection["polygon"] = polygon_xy.tolist()  # Variable-vertex polygon
                        detection["polygon_normalized"] = masks.xyn[i].tolist()  # Normalized [0-1]
                        detection["has_mask"] = True
                        logger.debug(f"Extracted polygon with {len(polygon_xy)} vertices")
                    else:
                        # Fall back to bbox corners if polygon is empty
                        x1, y1, x2, y2 = xyxy
                        detection["polygon"] = [
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                        ]
                        detection["has_mask"] = False
                else:
                    # Detection-only model: use bbox corners as polygon
                    x1, y1, x2, y2 = xyxy
                    detection["polygon"] = [
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ]
                    detection["has_mask"] = False
                
                detections.append(detection)
        
        # Log detection count
        mask_type = "segmented" if (results and results[0].masks is not None) else "detected"
        logger.info(f"{mask_type.capitalize()} {len(detections)} swimming pool(s)")
        
        return original_image, detections

    def detect_tiled(
        self,
        image_path: str,
        tile_size: int = 512,
        overlap: int = 64,
        image_size: int = 640
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run detection on large images using sliding window/tiling approach.
        
        This method splits large images into overlapping tiles, runs detection
        on each tile, and merges results using Non-Maximum Suppression.
        Essential for processing high-resolution aerial/satellite imagery
        where pools would be too small if the entire image was resized.
        
        Algorithm:
            1. Calculate grid of overlapping tiles
            2. Extract each tile from the original image
            3. Pad edge tiles if smaller than tile_size
            4. Run YOLO26 inference on each tile
            5. Map tile-local coordinates to global image coordinates
            6. Apply NMS to remove duplicate detections at tile boundaries
        
        Args:
            image_path (str): Path to the large input image.
            tile_size (int, optional): Width and height of each tile in pixels.
                Should match or be close to training image size (512 for this model).
                Defaults to 512.
            overlap (int, optional): Overlap between adjacent tiles in pixels.
                Larger overlap = better boundary detection but slower processing.
                Recommended: 10-20% of tile_size (e.g., 64 for 512 tiles).
                Defaults to 64.
            image_size (int, optional): YOLO inference size for each tile.
                Defaults to 640.
        
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Same format as detect()
                but coordinates are in global image space (not tile-local).
        
        Example:
            >>> # Process 10000x10000 pixel satellite image
            >>> image, detections = detector.detect_tiled(
            ...     "satellite.png",
            ...     tile_size=512,
            ...     overlap=64
            ... )
            >>> print(f"Found {len(detections)} pools in large image")
        
        Performance Notes:
            - A 10000x10000 image with 512 tile size and 64 overlap
              generates ~441 tiles (21x21 grid)
            - Processing time scales with number of tiles
            - GPU memory usage is constant (one tile at a time)
        """
        # Convert to Path object for robust path handling
        image_path = Path(image_path)
        
        # Validate image exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # -------------------------------------------------------------------
        # STEP 1: Load the original image
        # -------------------------------------------------------------------
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get image dimensions (height, width, channels)
        img_height, img_width = original_image.shape[:2]
        
        # -------------------------------------------------------------------
        # STEP 2: Check if tiling is actually needed
        # -------------------------------------------------------------------
        # If image is small enough (< 1.5x tile size), use standard detection
        if img_width <= tile_size * 1.5 and img_height <= tile_size * 1.5:
            logger.info("Image is small enough, using standard detection")
            return self.detect(image_path, image_size)
        
        logger.info(f"Large image detected ({img_width}x{img_height}), using tiled inference")
        logger.info(f"Tile size: {tile_size}, Overlap: {overlap}")
        
        # -------------------------------------------------------------------
        # STEP 3: Calculate tiling grid parameters
        # -------------------------------------------------------------------
        # Step size = tile_size - overlap (how far to move between tiles)
        step = tile_size - overlap
        
        # Calculate number of tiles in each dimension
        # Formula: ceil((dimension - overlap) / step)
        n_tiles_x = max(1, (img_width - overlap) // step + (1 if (img_width - overlap) % step else 0))
        n_tiles_y = max(1, (img_height - overlap) // step + (1 if (img_height - overlap) % step else 0))
        total_tiles = n_tiles_x * n_tiles_y
        
        logger.info(f"Processing {total_tiles} tiles ({n_tiles_x}x{n_tiles_y})")
        
        # -------------------------------------------------------------------
        # STEP 4: Process each tile
        # -------------------------------------------------------------------
        all_detections = []  # Accumulate detections from all tiles
        start_time = time.time()
        
        tile_count = 0
        # Iterate through tile positions (top-left corner of each tile)
        for y in range(0, img_height - overlap, step):
            for x in range(0, img_width - overlap, step):
                tile_count += 1
                
                # Calculate tile boundaries
                x1 = x                              # Left edge
                y1 = y                              # Top edge
                x2 = min(x + tile_size, img_width)  # Right edge (clipped)
                y2 = min(y + tile_size, img_height) # Bottom edge (clipped)
                
                # Extract tile from original image
                # NumPy slicing: [row_start:row_end, col_start:col_end]
                tile = original_image[y1:y2, x1:x2]
                
                # Pad tile if it's smaller than tile_size (edge tiles)
                # This ensures consistent input size for the model
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    # Create black padding canvas
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    # Copy tile to top-left corner
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                # Run inference on this tile
                results = self.model.predict(
                    source=tile,                  # Tile image array
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=image_size,
                    device=self.device,
                    verbose=False
                )
                
                # Process tile results and map to global coordinates
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        
                        # Get tile-local coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # ---------------------------------------------------
                        # Map tile coordinates to global image coordinates
                        # Global = tile_origin + local_coordinate
                        # ---------------------------------------------------
                        global_x1 = x1 + xyxy[0]  # Add tile's x offset
                        global_y1 = y1 + xyxy[1]  # Add tile's y offset
                        global_x2 = x1 + xyxy[2]
                        global_y2 = y1 + xyxy[3]
                        
                        # Clip coordinates to image boundaries
                        global_x1 = max(0, min(img_width, global_x1))
                        global_y1 = max(0, min(img_height, global_y1))
                        global_x2 = max(0, min(img_width, global_x2))
                        global_y2 = max(0, min(img_height, global_y2))
                        
                        # Create detection with global coordinates
                        detection = {
                            "bbox": [global_x1, global_y1, global_x2, global_y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": "swimming_pool"
                        }
                        all_detections.append(detection)
                
                # Progress logging every 50 tiles
                if tile_count % 50 == 0:
                    logger.info(f"Processed {tile_count}/{total_tiles} tiles...")
        
        # Calculate total inference time
        inference_time = time.time() - start_time
        logger.info(f"Tiled inference time: {inference_time * 1000:.2f} ms")
        
        # -------------------------------------------------------------------
        # STEP 5: Apply NMS to remove duplicate detections
        # -------------------------------------------------------------------
        # Overlapping tiles may detect the same pool multiple times
        # NMS merges these overlapping detections
        if all_detections:
            all_detections = self._apply_nms_to_detections(all_detections)
        
        logger.info(f"Detected {len(all_detections)} swimming pool(s) after NMS")
        
        return original_image, all_detections

    def _apply_nms_to_detections(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.
        
        NMS works by:
        1. Sort detections by confidence (highest first)
        2. Select the highest confidence detection
        3. Remove all detections with IoU > threshold with selected box
        4. Repeat until no detections remain
        
        Args:
            detections (List[Dict]): List of detection dictionaries.
                Each must have "bbox" and "confidence" keys.
            iou_threshold (float, optional): IoU threshold for merging.
                Boxes with IoU > threshold are considered duplicates.
                Defaults to 0.5.
        
        Returns:
            List[Dict]: Filtered detections with duplicates removed.
        
        Note:
            This is a custom implementation for post-processing tiled results.
            YOLO's built-in NMS handles within-image duplicates during inference.
        """
        # Handle empty input
        if not detections:
            return []
        
        # -------------------------------------------------------------------
        # STEP 1: Extract boxes and scores as numpy arrays for vectorized ops
        # -------------------------------------------------------------------
        boxes = np.array([d["bbox"] for d in detections])  # Shape: (N, 4)
        scores = np.array([d["confidence"] for d in detections])  # Shape: (N,)
        
        # Extract individual coordinates for IoU calculation
        x1 = boxes[:, 0]  # Left edge of all boxes
        y1 = boxes[:, 1]  # Top edge of all boxes
        x2 = boxes[:, 2]  # Right edge of all boxes
        y2 = boxes[:, 3]  # Bottom edge of all boxes
        
        # Calculate area of each box: width * height
        areas = (x2 - x1) * (y2 - y1)
        
        # -------------------------------------------------------------------
        # STEP 2: Sort by confidence (descending order)
        # -------------------------------------------------------------------
        order = scores.argsort()[::-1]  # Indices sorted by score (highest first)
        
        # -------------------------------------------------------------------
        # STEP 3: NMS loop - iteratively select best box and remove overlaps
        # -------------------------------------------------------------------
        keep = []  # Indices of boxes to keep
        
        while order.size > 0:
            # Select the box with highest confidence
            i = order[0]
            keep.append(i)
            
            # If only one box left, we're done
            if order.size == 1:
                break
            
            # Calculate IoU between selected box and all remaining boxes
            # Intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])  # Left edge of intersection
            yy1 = np.maximum(y1[i], y1[order[1:]])  # Top edge of intersection
            xx2 = np.minimum(x2[i], x2[order[1:]])  # Right edge of intersection
            yy2 = np.minimum(y2[i], y2[order[1:]])  # Bottom edge of intersection
            
            # Intersection dimensions (clamp to 0 if no overlap)
            w = np.maximum(0, xx2 - xx1)  # Intersection width
            h = np.maximum(0, yy2 - yy1)  # Intersection height
            
            # Intersection area
            intersection = w * h
            
            # IoU = intersection / union
            # Union = area_A + area_B - intersection
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep only boxes with IoU below threshold (non-overlapping)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 because we excluded order[0]
        
        # Return filtered detections
        return [detections[i] for i in keep]

    def detect_batch(
        self,
        image_dir: str,
        image_size: int = 640
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection on all images in a directory.
        
        Processes all supported image files in the directory and returns
        a dictionary mapping each image path to its detections.
        
        Args:
            image_dir (str): Path to directory containing images.
            image_size (int, optional): Inference image size. Defaults to 640.
        
        Returns:
            Dict[str, List[Dict]]: Dictionary where:
                - Keys are image file paths (str)
                - Values are lists of detection dictionaries
        
        Raises:
            ValueError: If image_dir is not a valid directory.
        
        Supported Formats:
            .jpg, .jpeg, .png, .bmp
        
        Example:
            >>> results = detector.detect_batch("images/")
            >>> for path, detections in results.items():
            ...     print(f"{path}: {len(detections)} pools")
        """
        # Convert to Path and validate
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Not a directory: {image_dir}")
        
        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        # Find all image files in directory
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        # Process each image
        all_detections = {}
        
        for image_path in image_files:
            try:
                # Run detection on single image
                _, detections = self.detect(str(image_path), image_size)
                all_detections[str(image_path)] = detections
            except Exception as e:
                # Log error but continue processing other images
                logger.error(f"Failed to process {image_path}: {e}")
        
        return all_detections


# =============================================================================
# IMAGE PROCESSING FUNCTION
# =============================================================================

def process_single_image(
    detector: PoolDetector,
    image_path: str,
    output_dir: str,
    image_size: int = 640,
    use_tiling: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 64
) -> None:
    """
    Process a single image and generate output files.
    
    This function orchestrates the full detection pipeline:
    1. Run detection (standard or tiled)
    2. Extract boundary coordinates for each pool
    3. Draw red outlines on detected pools
    4. Save coordinates.txt and output_image.jpg
    
    Args:
        detector (PoolDetector): Initialized detector instance.
        image_path (str): Path to input image file.
        output_dir (str): Directory to save output files.
            Created automatically if it doesn't exist.
        image_size (int, optional): YOLO inference size. Defaults to 640.
        use_tiling (bool, optional): Enable tiled inference for large images.
            Defaults to False.
        tile_size (int, optional): Tile size for tiled inference.
            Defaults to 512.
        tile_overlap (int, optional): Overlap between tiles.
            Defaults to 64.
    
    Output Files Created:
        - output_dir/output_image.jpg: Image with red pool outlines
        - output_dir/coordinates.txt: Pool coordinates (single detection)
        - output_dir/coordinates_0.txt, coordinates_1.txt, etc. (multiple)
    """
    # Import postprocessing functions from sibling module
    from inference.postprocessing import (
        draw_pool_outline,              # Draw polygon outline on image
        draw_pool_mask,                 # Draw filled semi-transparent mask
        extract_boundary_coordinates,   # Convert bbox to corner points
        extract_pool_contour,           # Extract actual pool contour using CV
        save_coordinates,               # Save coordinates to file
        save_segmentation_coordinates,  # Save polygon coordinates with shape
        classify_pool_shape,            # Classify pool shape
        save_as_geojson                 # Save as GeoJSON for maps
    )
    
    # Create output directory or clean it if it exists
    # This prevents coordinate files from previous runs from persisting
    output_dir = Path(output_dir)
    if output_dir.exists():
        # Remove old coordinates and images for this specific detection
        for f in output_dir.glob("coordinates*.txt"):
            f.unlink()
        for f in output_dir.glob("output_image.jpg"):
            f.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------
    # STEP 1: Run detection (choose method based on use_tiling flag)
    # -------------------------------------------------------------------
    if use_tiling:
        # Use tiled inference for large images
        original_image, detections = detector.detect_tiled(
            image_path, tile_size, tile_overlap, image_size
        )
    else:
        # Use standard inference for normal-sized images
        original_image, detections = detector.detect(image_path, image_size)

    
    # -------------------------------------------------------------------
    # STEP 2: Process each detection - Extract actual pool contours
    # -------------------------------------------------------------------
    logger.info(f"Detected {len(detections)} swimming pool(s)")
    
    # Store processed detections for GeoJSON export
    processed_detections = []
    
    for i, detection in enumerate(detections):
        confidence = detection["confidence"]
        has_mask = detection.get("has_mask", False)
        bbox = detection["bbox"]
        
        # Extract actual pool contour using color-based CV segmentation
        # This produces shape-accurate borders instead of rectangles
        polygon = extract_pool_contour(
            original_image,
            np.array(bbox),
            epsilon_factor=0.015,  # Smoother contour
            min_area_ratio=0.05
        )
        
        # Classify pool shape based on contour
        shape = classify_pool_shape(polygon)
        
        # Store for GeoJSON
        processed_detections.append({
            "polygon": polygon,
            "confidence": confidence,
            "shape": shape
        })
        
        # Draw pool contour on image (red outline + faint green mask)
        # Red outline is more visible and requested by user
        original_image = draw_pool_mask(
            original_image,
            polygon,
            color=(0, 0, 255),   # Red in BGR for outline
            alpha=0.2,           # Faint fill
            outline_thickness=3
        )
        
        # Generate coordinate filename
        suffix = f"_{i}" if len(detections) > 1 else ""
        coord_path = output_dir / f"coordinates{suffix}.txt"
        
        # Save contour coordinates with shape info
        save_segmentation_coordinates(polygon, str(coord_path), confidence, shape)
    
    # -------------------------------------------------------------------
    # STEP 3: Save output image and GeoJSON
    # -------------------------------------------------------------------
    output_image_path = output_dir / "output_image.jpg"
    cv2.imwrite(str(output_image_path), original_image)
    logger.info(f"Saved output image: {output_image_path}")
    
    # Save as GeoJSON for map visualization
    geojson_path = output_dir / "detections.geojson"
    try:
        origin_lat = getattr(detector, 'origin_lat', 43.7) # Default: Alpes-Maritimes
        origin_lng = getattr(detector, 'origin_lng', 7.2)
        mpp = getattr(detector, 'meters_per_pixel', 0.3)
        
        save_as_geojson(processed_detections, str(geojson_path), origin_lat, origin_lng, mpp)
        logger.info(f"Saved GeoJSON for map: {geojson_path}")
    except Exception as e:
        logger.warning(f"Could not save GeoJSON: {e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - input (str): Path to input image or directory
            - model (str): Path to model weights
            - output_dir (str): Output directory
            - conf_threshold (float): Confidence threshold
            - iou_threshold (float): IoU threshold for NMS
            - img_size (int): Inference image size
            - device (str): CUDA device or 'cpu'
            - tiled (bool): Enable tiled inference
            - tile_size (int): Tile size
            - tile_overlap (int): Tile overlap
    """
    # Create argument parser with description and help formatter
    parser = argparse.ArgumentParser(
        description="Detect swimming pools in aerial images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults
    )
    
    # -------------------------------------------------------------------------
    # Required Arguments
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images"
    )
    
    # -------------------------------------------------------------------------
    # Model and Output Arguments
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights (.pt file)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files (created if needed)"
    )
    
    # -------------------------------------------------------------------------
    # Detection Parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for detection (default: 0.35)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for non-maximum suppression (default: 0.45)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device ID (e.g., '0', '1') or 'cpu' for CPU inference"
    )
    
    # -------------------------------------------------------------------------
    # Tiling Arguments (for large images)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable sliding window/tiling for large images (10000x10000+)"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for tiled inference (should match training size)"
    )
    
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Overlap between tiles (pixels). More overlap = better boundary detection"
    )
    
    # -------------------------------------------------------------------------
    # Geospatial Arguments (for map mapping)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lat",
        type=float,
        default=43.7,
        help="Latitude of image top-left corner (default: Alpes-Maritimes region)"
    )
    
    parser.add_argument(
        "--lng",
        type=float,
        default=7.2,
        help="Longitude of image top-left corner"
    )
    
    parser.add_argument(
        "--mpp",
        type=float,
        default=0.3,
        help="Meters per pixel resolution (default: 0.3m/px)"
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main entry point for the detection script.
    
    This function:
    1. Parses command line arguments
    2. Initializes the PoolDetector
    3. Processes input (single image or directory)
    4. Saves output files to specified directory
    
    Usage:
        # Standard detection
        python detect_pools.py --input image.jpg
        
        # Tiled detection for large images
        python detect_pools.py --input large.png --tiled
        
        # Custom settings
        python detect_pools.py --input img.jpg --conf-threshold 0.7 --device cpu
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle empty default values (for backwards compatibility)
    if args.model == "":
        args.model = "weights/best.pt"      

    if args.output_dir == "":
        args.output_dir = "output"      

    # -------------------------------------------------------------------------
    # Initialize the detector
    # -------------------------------------------------------------------------
    detector = PoolDetector(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # Attach geospatial settings to detector so process_single_image can access them
    detector.origin_lat = args.lat
    detector.origin_lng = args.lng
    detector.meters_per_pixel = args.mpp
    
    # -------------------------------------------------------------------------
    # Process input (file or directory)
    # -------------------------------------------------------------------------
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image mode
        process_single_image(
            detector,
            str(input_path),
            args.output_dir,
            args.img_size,
            use_tiling=args.tiled,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap
        )
    elif input_path.is_dir():
        # Directory mode - process all images
        for img_file in input_path.iterdir():
            # Filter for supported image extensions
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                # Create subdirectory for each image's outputs
                img_output_dir = Path(args.output_dir) / img_file.stem
                try:
                    process_single_image(
                        detector,
                        str(img_file),
                        str(img_output_dir),
                        args.img_size,
                        use_tiling=args.tiled,
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap
                    )
                except Exception as e:
                    # Log error but continue with other images
                    logger.error(f"Failed to process {img_file}: {e}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Standard Python idiom: only run main() if this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
