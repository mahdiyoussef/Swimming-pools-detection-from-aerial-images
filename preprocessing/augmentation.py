"""
Data Augmentation Module for Swimming Pool Detection System.

This module provides data augmentation pipelines using Albumentations
library for training the swimming pool detection model.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Data augmentation pipeline for YOLO26 training.

    This class provides configurable augmentation transforms
    optimized for aerial swimming pool detection.

    Attributes:
        config: Augmentation configuration dictionary.
        transform: Albumentations Compose transform.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the AugmentationPipeline.

        Args:
            config_path: Path to the dataset configuration file.
        """
        self.project_root = Path(__file__).resolve().parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "dataset_config.yaml"
        
        self.config = self._load_config(config_path)
        self.aug_config = self.config.get("augmentation", {})
        
        # Build transform pipeline if augmentation is enabled
        if self.aug_config.get("enabled", True):
            self.transform = self._build_transform()
        else:
            self.transform = None

    def _load_config(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            return {"augmentation": {"enabled": True}}
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_transform(self) -> A.Compose:
        """
        Build Albumentations transform pipeline.

        Returns:
            A.Compose: Composed augmentation transforms.
        """
        transforms = []
        
        # Horizontal flip
        if self.aug_config.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        # Vertical flip
        if self.aug_config.get("vertical_flip", True):
            transforms.append(A.VerticalFlip(p=0.5))
        
        # Random rotation
        rotation_limit = self.aug_config.get("rotation_limit", 15)
        if rotation_limit > 0:
            transforms.append(
                A.Rotate(
                    limit=rotation_limit,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                )
            )
        
        # Brightness and contrast
        brightness_limit = self.aug_config.get("brightness_limit", 0.2)
        contrast_limit = self.aug_config.get("contrast_limit", 0.2)
        if brightness_limit > 0 or contrast_limit > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=0.5
                )
            )
        
        # Random scaling
        scale_limit = self.aug_config.get("scale_limit", 0.2)
        if scale_limit > 0:
            transforms.append(
                A.RandomScale(
                    scale_limit=scale_limit,
                    p=0.5
                )
            )
        
        # Additional useful augmentations for aerial imagery
        transforms.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ])
        
        # Compose with bounding box support
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3
            )
        )

    def apply(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply augmentation to image and bounding boxes.

        Args:
            image: Input image as numpy array (H, W, C).
            bboxes: List of bounding boxes in YOLO format
                   [x_center, y_center, width, height].
            class_labels: List of class labels for each bbox.

        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_labels).
        """
        if self.transform is None:
            return image, bboxes, class_labels
        
        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                transformed["image"],
                transformed["bboxes"],
                transformed["class_labels"]
            )
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Returning original.")
            return image, bboxes, class_labels

    def get_training_transform(
        self,
        image_size: int = 640
    ) -> A.Compose:
        """
        Get training augmentation with resize.

        Args:
            image_size: Target image size.

        Returns:
            A.Compose: Training transform pipeline.
        """
        base_transforms = self._build_transform().transforms.copy()
        
        # Add resize at the end
        base_transforms.append(A.Resize(image_size, image_size))
        
        return A.Compose(
            base_transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3
            )
        )

    def get_validation_transform(
        self,
        image_size: int = 640
    ) -> A.Compose:
        """
        Get validation transform (resize only).

        Args:
            image_size: Target image size.

        Returns:
            A.Compose: Validation transform pipeline.
        """
        return A.Compose(
            [A.Resize(image_size, image_size)],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3
            )
        )


def load_yolo_labels(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Load YOLO format labels from file.

    Args:
        label_path: Path to the label file.

    Returns:
        Tuple of (bboxes, class_labels).
    """
    bboxes = []
    class_labels = []
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
    
    return bboxes, class_labels


def save_yolo_labels(
    label_path: Path,
    bboxes: List[List[float]],
    class_labels: List[int]
) -> None:
    """
    Save labels in YOLO format.

    Args:
        label_path: Path to save the label file.
        bboxes: List of bounding boxes.
        class_labels: List of class labels.
    """
    with open(label_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


def augment_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    num_augmentations: int = 3,
    config_path: Optional[str] = None
) -> None:
    """
    Augment an entire dataset.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing labels.
        output_images_dir: Output directory for augmented images.
        output_labels_dir: Output directory for augmented labels.
        num_augmentations: Number of augmentations per image.
        config_path: Path to configuration file.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = AugmentationPipeline(config_path=config_path)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        # Load image and labels
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = load_yolo_labels(label_path)
        
        # Copy original
        cv2.imwrite(
            str(output_images_dir / img_path.name),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        save_yolo_labels(
            output_labels_dir / f"{img_path.stem}.txt",
            bboxes,
            class_labels
        )
        
        # Generate augmentations
        for i in range(num_augmentations):
            aug_image, aug_bboxes, aug_labels = pipeline.apply(
                image, bboxes, class_labels
            )
            
            if len(aug_bboxes) > 0:  # Only save if bboxes remain
                aug_name = f"{img_path.stem}_aug{i}"
                cv2.imwrite(
                    str(output_images_dir / f"{aug_name}{img_path.suffix}"),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                save_yolo_labels(
                    output_labels_dir / f"{aug_name}.txt",
                    aug_bboxes,
                    aug_labels
                )
    
    logger.info(f"Augmentation complete. Output saved to: {output_images_dir}")


def main() -> None:
    """Main entry point for augmentation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply data augmentation to dataset"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Directory containing labels"
    )
    parser.add_argument(
        "--output-images",
        type=str,
        required=True,
        help="Output directory for augmented images"
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        required=True,
        help="Output directory for augmented labels"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        help="Number of augmentations per image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    augment_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_images_dir=Path(args.output_images),
        output_labels_dir=Path(args.output_labels),
        num_augmentations=args.num_augmentations,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
