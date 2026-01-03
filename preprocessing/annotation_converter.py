"""
Annotation Converter Module for Swimming Pool Detection System.

This module provides functionality to convert annotations from various
formats (COCO JSON, Pascal VOC XML) to YOLO format.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class AnnotationConverter:
    """
    Convert annotations from various formats to YOLO format.

    Supports conversion from:
    - COCO JSON format
    - Pascal VOC XML format
    - Polygon coordinates to bounding boxes

    Attributes:
        class_mapping: Dictionary mapping class names to class IDs.
    """

    def __init__(self, class_mapping: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize the AnnotationConverter.

        Args:
            class_mapping: Dictionary mapping class names to class IDs.
                          If None, defaults to {"swimming_pool": 0}.
        """
        if class_mapping is None:
            class_mapping = {"swimming_pool": 0}
        
        self.class_mapping = class_mapping
        logger.info(f"Initialized with classes: {list(class_mapping.keys())}")

    def coco_to_yolo(
        self,
        coco_json_path: Path,
        output_dir: Path,
        images_dir: Optional[Path] = None
    ) -> int:
        """
        Convert COCO JSON annotations to YOLO format.

        Args:
            coco_json_path: Path to COCO JSON file.
            output_dir: Directory to save YOLO label files.
            images_dir: Optional directory containing images (for verification).

        Returns:
            int: Number of annotations converted.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
        
        # Build image ID to info mapping
        images_info = {img["id"]: img for img in coco_data["images"]}
        
        # Build category ID to name mapping
        category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        # Group annotations by image
        annotations_by_image: Dict[int, List] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        converted_count = 0
        
        for img_id, annotations in annotations_by_image.items():
            if img_id not in images_info:
                continue
            
            img_info = images_info[img_id]
            img_width = img_info["width"]
            img_height = img_info["height"]
            img_filename = img_info["file_name"]
            
            # Create YOLO label file
            label_filename = Path(img_filename).stem + ".txt"
            label_path = output_dir / label_filename
            
            yolo_lines = []
            
            for ann in annotations:
                category_name = category_mapping.get(ann["category_id"], "unknown")
                
                if category_name not in self.class_mapping:
                    continue
                
                class_id = self.class_mapping[category_name]
                
                # COCO bbox: [x_min, y_min, width, height]
                bbox = ann["bbox"]
                
                # Convert to YOLO format
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                converted_count += 1
            
            # Write YOLO label file
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
        
        logger.info(f"Converted {converted_count} annotations from COCO format")
        return converted_count

    def voc_to_yolo(
        self,
        xml_path: Path,
        output_path: Path,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None
    ) -> int:
        """
        Convert Pascal VOC XML annotation to YOLO format.

        Args:
            xml_path: Path to VOC XML file.
            output_path: Path to save YOLO label file.
            img_width: Image width (extracted from XML if not provided).
            img_height: Image height (extracted from XML if not provided).

        Returns:
            int: Number of annotations converted.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions from XML if not provided
        if img_width is None or img_height is None:
            size_elem = root.find("size")
            if size_elem is not None:
                img_width = int(size_elem.find("width").text)
                img_height = int(size_elem.find("height").text)
            else:
                raise ValueError("Image dimensions not found in XML and not provided")
        
        yolo_lines = []
        converted_count = 0
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            
            if class_name not in self.class_mapping:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            class_id = self.class_mapping[class_name]
            
            bndbox = obj.find("bndbox")
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)
            
            # Convert to YOLO format
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Clamp values
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            converted_count += 1
        
        # Write YOLO label file
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

        Args:
            xml_dir: Directory containing VOC XML files.
            output_dir: Directory to save YOLO label files.

        Returns:
            int: Total number of annotations converted.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        total_converted = 0
        
        for xml_path in xml_dir.glob("*.xml"):
            output_path = output_dir / f"{xml_path.stem}.txt"
            try:
                count = self.voc_to_yolo(xml_path, output_path)
                total_converted += count
            except Exception as e:
                logger.error(f"Failed to convert {xml_path}: {e}")
        
        logger.info(f"Converted {total_converted} annotations from VOC format")
        return total_converted

    @staticmethod
    def polygon_to_bbox(
        polygon: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Convert polygon coordinates to bounding box.

        Args:
            polygon: List of (x, y) coordinate tuples.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max).
        """
        if not polygon:
            raise ValueError("Empty polygon provided")
        
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
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
        Convert polygon to YOLO format annotation.

        Args:
            polygon: List of (x, y) coordinate tuples (absolute pixels).
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            class_id: Class ID for the annotation.

        Returns:
            str: YOLO format annotation line.
        """
        x_min, y_min, x_max, y_max = self.polygon_to_bbox(polygon)
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Clamp values
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    @staticmethod
    def yolo_to_absolute(
        yolo_bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert YOLO normalized bbox to absolute pixel coordinates.

        Args:
            yolo_bbox: Tuple of (x_center, y_center, width, height) normalized.
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in absolute pixels.
        """
        x_center, y_center, width, height = yolo_bbox
        
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        x_min = int(x_center_abs - width_abs / 2)
        y_min = int(y_center_abs - height_abs / 2)
        x_max = int(x_center_abs + width_abs / 2)
        y_max = int(y_center_abs + height_abs / 2)
        
        return x_min, y_min, x_max, y_max

    @staticmethod
    def validate_yolo_annotation(label_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a YOLO format annotation file.

        Args:
            label_path: Path to the YOLO label file.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        
        if not label_path.exists():
            return False, ["File does not exist"]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if len(parts) < 5:
                errors.append(f"Line {i + 1}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                if class_id < 0:
                    errors.append(f"Line {i + 1}: Class ID must be non-negative")
                
                for j, name in enumerate(["x_center", "y_center", "width", "height"]):
                    value = float(parts[j + 1])
                    if value < 0 or value > 1:
                        errors.append(
                            f"Line {i + 1}: {name}={value} not in [0, 1]"
                        )
            except ValueError as e:
                errors.append(f"Line {i + 1}: Invalid number format - {e}")
        
        return len(errors) == 0, errors


def main() -> None:
    """Main entry point for annotation conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert annotations to YOLO format"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "voc"],
        required=True,
        help="Source annotation format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for YOLO labels"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=["swimming_pool"],
        help="Class names in order of class ID"
    )
    
    args = parser.parse_args()
    
    # Build class mapping
    class_mapping = {name: i for i, name in enumerate(args.class_names)}
    converter = AnnotationConverter(class_mapping=class_mapping)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.format == "coco":
        converter.coco_to_yolo(input_path, output_path)
    elif args.format == "voc":
        if input_path.is_dir():
            converter.voc_dir_to_yolo(input_path, output_path)
        else:
            converter.voc_to_yolo(input_path, output_path / f"{input_path.stem}.txt")


if __name__ == "__main__":
    main()
