"""
Annotation Converter Module for Swimming Pool Detection System.
================================================================

This module provides functionality to convert annotations between various
formats commonly used in object detection. The primary target format is
YOLO, which is required by the YOLO26 model.

Supported Source Formats:
    - COCO JSON: Common format from annotation tools like LabelMe, CVAT
    - Pascal VOC XML: Earlier standard from the PASCAL VOC challenge
    - Polygon coordinates: Direct conversion from polygon points

Target Format (YOLO):
    Each .txt file contains one line per object:
    <class_id> <x_center> <y_center> <width> <height>
    
    Where all coordinates are normalized to [0, 1]:
    - x_center = center_x / image_width
    - y_center = center_y / image_height
    - width = bbox_width / image_width
    - height = bbox_height / image_height

Format Comparison:
    +------------------+----------------------------------+
    | Format           | Bounding Box Representation      |
    +------------------+----------------------------------+
    | COCO             | [x_min, y_min, width, height]    |
    | Pascal VOC       | [xmin, ymin, xmax, ymax]         |
    | YOLO             | [x_center, y_center, w, h] norm. |
    +------------------+----------------------------------+

Usage:
    # Convert from VOC XML
    converter = AnnotationConverter({"pool": 0})
    converter.voc_to_yolo("annotation.xml", "labels/annotation.txt")
    
    # Convert from COCO JSON
    converter.coco_to_yolo("annotations.json", "labels/")

Author: Swimming Pool Detection Team
Date: 2026-01-02
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json                        # JSON parsing for COCO format
import logging                     # Logging functionality
import xml.etree.ElementTree as ET # XML parsing for Pascal VOC format
from pathlib import Path           # Cross-platform file path handling
from typing import Dict, List, Optional, Tuple  # Type hints

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,                                    # Log INFO and above
    format="[%(asctime)s] [%(levelname)s] %(message)s",   # Log format
    datefmt="%Y-%m-%d %H:%M:%S"                           # Timestamp format
)
logger = logging.getLogger(__name__)  # Get logger for this module


# =============================================================================
# ANNOTATION CONVERTER CLASS
# =============================================================================

class AnnotationConverter:
    """
    Convert annotations from various formats to YOLO format.
    
    This class provides methods to convert annotation files from COCO JSON
    and Pascal VOC XML formats to YOLO's normalized text format. It also
    provides utility functions for coordinate transformations.
    
    Supported conversions:
        - COCO JSON -> YOLO TXT
        - Pascal VOC XML -> YOLO TXT
        - Polygon coordinates -> Bounding box -> YOLO format
        - YOLO normalized -> Absolute pixel coordinates
    
    Attributes:
        class_mapping (Dict[str, int]): Dictionary mapping class names to
            their corresponding class IDs (0-indexed). The class ID determines
            the first value in each YOLO annotation line.
    
    Example:
        >>> # Single class (swimming pools)
        >>> converter = AnnotationConverter({"swimming_pool": 0})
        >>> 
        >>> # Multi-class detection
        >>> converter = AnnotationConverter({
        ...     "swimming_pool": 0,
        ...     "building": 1,
        ...     "road": 2
        ... })
    """

    def __init__(self, class_mapping: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize the AnnotationConverter with class mapping.
        
        Args:
            class_mapping (Optional[Dict[str, int]]): Dictionary mapping class
                names (as they appear in source annotations) to class IDs
                (integers starting from 0).
                
                Example: {"swimming_pool": 0, "tennis_court": 1}
                
                If None, defaults to {"swimming_pool": 0} for single-class
                pool detection.
        
        Note:
            Class names must match exactly what appears in the source
            annotation files (case-sensitive for VOC XML, may vary for COCO).
        """
        # Use default mapping if none provided
        # This is the standard mapping for swimming pool detection
        if class_mapping is None:
            class_mapping = {"swimming_pool": 0}
        
        # Store the class mapping as instance attribute
        self.class_mapping = class_mapping
        
        # Log the initialized classes for debugging
        logger.info(f"Initialized with classes: {list(class_mapping.keys())}")

    # =========================================================================
    # COCO FORMAT CONVERSION
    # =========================================================================

    def coco_to_yolo(
        self,
        coco_json_path: Path,
        output_dir: Path,
        images_dir: Optional[Path] = None
    ) -> int:
        """
        Convert COCO JSON annotations to YOLO format.
        
        COCO format uses a single JSON file containing all annotations for
        all images. This method extracts annotations for each image and
        creates individual YOLO .txt files.
        
        COCO Bounding Box Format:
            [x_min, y_min, width, height] in absolute pixels
        
        YOLO Bounding Box Format:
            [x_center, y_center, width, height] normalized to [0, 1]
        
        Conversion Formula:
            x_center = (x_min + width/2) / image_width
            y_center = (y_min + height/2) / image_height
            width_norm = width / image_width
            height_norm = height / image_height
        
        Args:
            coco_json_path (Path): Path to the COCO JSON annotation file.
                File should contain 'images', 'annotations', and 'categories'.
            output_dir (Path): Directory to save YOLO label files (.txt).
                Created automatically if it doesn't exist.
                One file per image: {image_stem}.txt
            images_dir (Optional[Path]): Directory containing images.
                Currently unused, reserved for future image verification.
        
        Returns:
            int: Total number of individual annotations converted.
                (Not the number of files, but the number of objects)
        
        COCO JSON Structure:
            {
                "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x,y,w,h]}],
                "categories": [{"id": 1, "name": "swimming_pool"}]
            }
        
        Example:
            >>> converter = AnnotationConverter({"swimming_pool": 0})
            >>> count = converter.coco_to_yolo(
            ...     Path("annotations.json"),
            ...     Path("labels/")
            ... )
            >>> print(f"Converted {count} annotations")
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # -------------------------------------------------------------------
        # STEP 1: Load and parse COCO JSON file
        # -------------------------------------------------------------------
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
        
        # -------------------------------------------------------------------
        # STEP 2: Build lookup dictionaries for efficient access
        # -------------------------------------------------------------------
        
        # Map image ID to image info (filename, dimensions)
        # This allows O(1) lookup when processing annotations
        images_info = {img["id"]: img for img in coco_data["images"]}
        
        # Map category ID to category name
        # COCO uses numeric category IDs in annotations
        category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        # -------------------------------------------------------------------
        # STEP 3: Group annotations by image
        # -------------------------------------------------------------------
        # This is more efficient than iterating through all annotations
        # for each image (O(n) vs O(n*m))
        annotations_by_image: Dict[int, List] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # -------------------------------------------------------------------
        # STEP 4: Convert annotations for each image
        # -------------------------------------------------------------------
        converted_count = 0
        
        for img_id, annotations in annotations_by_image.items():
            # Skip if image info not found
            if img_id not in images_info:
                continue
            
            # Get image info for dimensions
            img_info = images_info[img_id]
            img_width = img_info["width"]
            img_height = img_info["height"]
            img_filename = img_info["file_name"]
            
            # Create output filename (same stem as image, .txt extension)
            label_filename = Path(img_filename).stem + ".txt"
            label_path = output_dir / label_filename
            
            # Process each annotation for this image
            yolo_lines = []
            
            for ann in annotations:
                # Get category name from ID
                category_name = category_mapping.get(ann["category_id"], "unknown")
                
                # Skip if category not in our class mapping
                if category_name not in self.class_mapping:
                    continue
                
                # Get our class ID for this category
                class_id = self.class_mapping[category_name]
                
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann["bbox"]
                
                # ----------------------------------------------------------
                # Convert COCO to YOLO format
                # COCO: [x_min, y_min, width, height] in pixels
                # YOLO: [x_center, y_center, width, height] normalized
                # ----------------------------------------------------------
                
                # Calculate center from top-left corner
                x_center = (bbox[0] + bbox[2] / 2) / img_width   # x_min + width/2
                y_center = (bbox[1] + bbox[3] / 2) / img_height  # y_min + height/2
                
                # Normalize dimensions
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Clamp values to [0, 1] range (handle edge cases)
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Format as YOLO line with 6 decimal precision
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                converted_count += 1
            
            # Write YOLO label file (one line per object)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
        
        # Log summary
        logger.info(f"Converted {converted_count} annotations from COCO format")
        return converted_count

    # =========================================================================
    # PASCAL VOC FORMAT CONVERSION
    # =========================================================================

    def voc_to_yolo(
        self,
        xml_path: Path,
        output_path: Path,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None
    ) -> int:
        """
        Convert a single Pascal VOC XML annotation to YOLO format.
        
        Pascal VOC format uses one XML file per image, containing object
        bounding boxes in absolute pixel coordinates as [xmin, ymin, xmax, ymax].
        
        Pascal VOC Bounding Box Format:
            <bndbox>
                <xmin>100</xmin>  <!-- Left edge -->
                <ymin>50</ymin>   <!-- Top edge -->
                <xmax>200</xmax>  <!-- Right edge -->
                <ymax>150</ymax>  <!-- Bottom edge -->
            </bndbox>
        
        Conversion Formula:
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
        
        Args:
            xml_path (Path): Path to Pascal VOC XML file.
            output_path (Path): Path to save YOLO label file (.txt).
            img_width (Optional[int]): Image width in pixels.
                If None, extracted from XML <size> element.
            img_height (Optional[int]): Image height in pixels.
                If None, extracted from XML <size> element.
        
        Returns:
            int: Number of objects converted from this file.
        
        Raises:
            ValueError: If image dimensions not found and not provided.
        
        Example:
            >>> converter = AnnotationConverter({"pool": 0})
            >>> count = converter.voc_to_yolo(
            ...     Path("annotations/image1.xml"),
            ...     Path("labels/image1.txt")
            ... )
        """
        # -------------------------------------------------------------------
        # STEP 1: Parse XML file
        # -------------------------------------------------------------------
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # -------------------------------------------------------------------
        # STEP 2: Get image dimensions
        # -------------------------------------------------------------------
        # Priority: provided arguments > XML <size> element
        if img_width is None or img_height is None:
            size_elem = root.find("size")
            if size_elem is not None:
                img_width = int(size_elem.find("width").text)
                img_height = int(size_elem.find("height").text)
            else:
                raise ValueError("Image dimensions not found in XML and not provided")
        
        # -------------------------------------------------------------------
        # STEP 3: Process each object in the annotation
        # -------------------------------------------------------------------
        yolo_lines = []
        converted_count = 0
        
        # Find all <object> elements in XML
        for obj in root.findall("object"):
            # Get class name from <name> element
            class_name = obj.find("name").text
            
            # Skip unknown classes (not in our mapping)
            if class_name not in self.class_mapping:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            # Get our class ID
            class_id = self.class_mapping[class_name]
            
            # Get bounding box coordinates from <bndbox> element
            bndbox = obj.find("bndbox")
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)
            
            # ----------------------------------------------------------
            # Convert VOC to YOLO format
            # VOC: [xmin, ymin, xmax, ymax] in pixels
            # YOLO: [x_center, y_center, width, height] normalized
            # ----------------------------------------------------------
            
            # Calculate center coordinates
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            
            # Calculate normalized dimensions
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Clamp values to valid range [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Format as YOLO line
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            converted_count += 1
        
        # -------------------------------------------------------------------
        # STEP 4: Write YOLO label file
        # -------------------------------------------------------------------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        return converted_count

    def voc_dir_to_yolo(
        self,
        xml_dir: Path,
        output_dir: Path
    ) -> int:
        """
        Convert all Pascal VOC XML files in a directory to YOLO format.
        
        Batch conversion of an entire directory of VOC annotations.
        Each .xml file is converted to a corresponding .txt file.
        
        Args:
            xml_dir (Path): Directory containing .xml annotation files.
            output_dir (Path): Directory to save .txt YOLO label files.
                Created automatically if it doesn't exist.
        
        Returns:
            int: Total number of objects converted across all files.
        
        Note:
            Errors in individual files are logged but don't stop processing.
        
        Example:
            >>> converter = AnnotationConverter({"pool": 0})
            >>> total = converter.voc_dir_to_yolo(
            ...     Path("voc_annotations/"),
            ...     Path("yolo_labels/")
            ... )
            >>> print(f"Converted {total} objects total")
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_converted = 0
        
        # Process each XML file in the directory
        for xml_path in xml_dir.glob("*.xml"):
            # Create output path with same stem but .txt extension
            output_path = output_dir / f"{xml_path.stem}.txt"
            
            try:
                # Convert single file
                count = self.voc_to_yolo(xml_path, output_path)
                total_converted += count
            except Exception as e:
                # Log error but continue with other files
                logger.error(f"Failed to convert {xml_path}: {e}")
        
        logger.info(f"Converted {total_converted} annotations from VOC format")
        return total_converted

    # =========================================================================
    # POLYGON CONVERSION UTILITIES
    # =========================================================================

    @staticmethod
    def polygon_to_bbox(
        polygon: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Convert polygon coordinates to axis-aligned bounding box.
        
        Computes the minimum bounding rectangle that contains all polygon
        points. Useful for converting segmentation masks to bounding boxes.
        
        Args:
            polygon (List[Tuple[float, float]]): List of (x, y) coordinate
                tuples forming the polygon vertices.
        
        Returns:
            Tuple[float, float, float, float]: Bounding box as
                (x_min, y_min, x_max, y_max) in same units as input.
        
        Raises:
            ValueError: If polygon is empty.
        
        Example:
            >>> polygon = [(100, 50), (150, 75), (120, 150)]
            >>> bbox = AnnotationConverter.polygon_to_bbox(polygon)
            >>> print(bbox)
            (100, 50, 150, 150)
        """
        # Validate input
        if not polygon:
            raise ValueError("Empty polygon provided")
        
        # Extract x and y coordinates separately
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        # Compute bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return x_min, y_min, x_max, y_max

    def polygon_to_yolo(
        self,
        polygon: List[Tuple[float, float]],
        img_width: int,
        img_height: int,
        class_id: int = 0
    ) -> str:
        """
        Convert polygon coordinates directly to YOLO format annotation.
        
        Convenience method that combines polygon_to_bbox() and normalization
        in a single call.
        
        Args:
            polygon (List[Tuple[float, float]]): List of (x, y) polygon
                vertices in absolute pixel coordinates.
            img_width (int): Image width in pixels (for normalization).
            img_height (int): Image height in pixels (for normalization).
            class_id (int, optional): Class ID for the annotation.
                Defaults to 0.
        
        Returns:
            str: YOLO format annotation line:
                "<class_id> <x_center> <y_center> <width> <height>"
        
        Example:
            >>> polygon = [(100, 50), (200, 50), (200, 150), (100, 150)]
            >>> line = converter.polygon_to_yolo(polygon, 640, 480)
            >>> print(line)
            "0 0.234375 0.208333 0.156250 0.208333"
        """
        # Get bounding box from polygon
        x_min, y_min, x_max, y_max = self.polygon_to_bbox(polygon)
        
        # Convert to YOLO format (same formula as voc_to_yolo)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Clamp to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # =========================================================================
    # YOLO FORMAT UTILITIES
    # =========================================================================

    @staticmethod
    def yolo_to_absolute(
        yolo_bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert YOLO normalized bbox to absolute pixel coordinates.
        
        Inverse of the YOLO normalization. Converts normalized center-based
        coordinates back to absolute pixel coordinates in [xmin, ymin, xmax, ymax]
        format.
        
        Conversion Formula:
            x_min = (x_center - width/2) * img_width
            y_min = (y_center - height/2) * img_height
            x_max = (x_center + width/2) * img_width
            y_max = (y_center + height/2) * img_height
        
        Args:
            yolo_bbox (Tuple[float, float, float, float]): YOLO format bbox
                as (x_center, y_center, width, height) all in [0, 1].
            img_width (int): Image width in pixels.
            img_height (int): Image height in pixels.
        
        Returns:
            Tuple[int, int, int, int]: Absolute pixel coordinates as
                (x_min, y_min, x_max, y_max).
        
        Example:
            >>> yolo_bbox = (0.5, 0.5, 0.2, 0.3)  # Center box
            >>> abs_bbox = AnnotationConverter.yolo_to_absolute(
            ...     yolo_bbox, 640, 480
            ... )
            >>> print(abs_bbox)
            (256, 168, 384, 312)
        """
        # Unpack normalized values
        x_center, y_center, width, height = yolo_bbox
        
        # Convert to absolute coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Calculate corner coordinates
        x_min = int(x_center_abs - width_abs / 2)
        y_min = int(y_center_abs - height_abs / 2)
        x_max = int(x_center_abs + width_abs / 2)
        y_max = int(y_center_abs + height_abs / 2)
        
        return x_min, y_min, x_max, y_max

    @staticmethod
    def validate_yolo_annotation(label_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a YOLO format annotation file for correctness.
        
        Checks that the label file follows YOLO format requirements:
        - Each line has exactly 5 space-separated values
        - First value is a non-negative integer (class ID)
        - Values 2-5 are floats in range [0, 1]
        
        Args:
            label_path (Path): Path to the YOLO label file (.txt).
        
        Returns:
            Tuple[bool, List[str]]: A tuple containing:
                - bool: True if file is valid, False otherwise
                - List[str]: List of error messages (empty if valid)
        
        Example:
            >>> is_valid, errors = AnnotationConverter.validate_yolo_annotation(
            ...     Path("labels/image1.txt")
            ... )
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(error)
        """
        errors = []
        
        # Check file exists
        if not label_path.exists():
            return False, ["File does not exist"]
        
        # Read all lines
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        # Validate each line
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Split into components
            parts = line.split()
            
            # Check number of values
            if len(parts) < 5:
                errors.append(f"Line {i + 1}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                # Validate class ID (first value)
                class_id = int(parts[0])
                if class_id < 0:
                    errors.append(f"Line {i + 1}: Class ID must be non-negative")
                
                # Validate bbox values (must be in [0, 1])
                value_names = ["x_center", "y_center", "width", "height"]
                for j, name in enumerate(value_names):
                    value = float(parts[j + 1])
                    if value < 0 or value > 1:
                        errors.append(
                            f"Line {i + 1}: {name}={value} not in [0, 1]"
                        )
            except ValueError as e:
                errors.append(f"Line {i + 1}: Invalid number format - {e}")
        
        # Return validation result
        return len(errors) == 0, errors


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main() -> None:
    """
    Main entry point for command-line annotation conversion.
    
    Provides a CLI for converting annotations between formats.
    
    Usage:
        # Convert VOC directory to YOLO
        python annotation_converter.py --format voc --input voc/ --output labels/
        
        # Convert COCO JSON to YOLO
        python annotation_converter.py --format coco --input annotations.json --output labels/
        
        # Custom class names
        python annotation_converter.py --format voc --input voc/ --output labels/ \
            --class-names pool building road
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Convert annotations to YOLO format"
    )
    
    # Source format argument
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "voc"],
        required=True,
        help="Source annotation format (coco or voc)"
    )
    
    # Input path argument
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (COCO JSON) or directory (VOC XMLs)"
    )
    
    # Output directory argument
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for YOLO label files"
    )
    
    # Class names argument
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=["swimming_pool"],
        help="Class names in order of class ID (default: swimming_pool)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build class mapping from class names
    # Index in list becomes class ID: ["pool", "court"] -> {"pool": 0, "court": 1}
    class_mapping = {name: i for i, name in enumerate(args.class_names)}
    
    # Create converter instance
    converter = AnnotationConverter(class_mapping=class_mapping)
    
    # Convert paths to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Execute conversion based on format
    if args.format == "coco":
        converter.coco_to_yolo(input_path, output_path)
    elif args.format == "voc":
        if input_path.is_dir():
            # Convert entire directory
            converter.voc_dir_to_yolo(input_path, output_path)
        else:
            # Convert single file
            converter.voc_to_yolo(input_path, output_path / f"{input_path.stem}.txt")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
