"""
Dataset Splitting Module for Swimming Pool Detection System.

This module handles the splitting of dataset into train, validation,
and test sets with optional stratification.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Split dataset into train, validation, and test sets.

    This class handles the complete workflow of splitting image-label pairs
    while maintaining data integrity and generating YOLO26-compatible
    dataset configuration.

    Attributes:
        config: Dataset configuration dictionary.
        project_root: Root path of the project.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        random_seed: Random seed for reproducibility.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the DatasetSplitter.

        Args:
            config_path: Path to the dataset configuration file.
        """
        self.project_root = Path(__file__).resolve().parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "dataset_config.yaml"
        
        self.config = self._load_config(config_path)
        
        # Extract split ratios
        splits = self.config["splits"]
        self.train_ratio = splits["train_ratio"]
        self.val_ratio = splits["val_ratio"]
        self.test_ratio = splits["test_ratio"]
        self.random_seed = splits["random_seed"]
        self.stratify = splits.get("stratify", False)
        
        # Verify ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

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
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config

    def find_image_label_pairs(
        self,
        images_dir: Path,
        labels_dir: Path
    ) -> List[Tuple[Path, Path]]:
        """
        Find matching image and label file pairs.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing label files.

        Returns:
            List of (image_path, label_path) tuples.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        pairs = []
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                logger.warning(f"No label found for image: {img_path.name}")
        
        logger.info(f"Found {len(pairs)} image-label pairs")
        return pairs

    def get_class_distribution(
        self,
        pairs: List[Tuple[Path, Path]]
    ) -> Dict[int, List[Tuple[Path, Path]]]:
        """
        Get class distribution for stratified splitting.

        Args:
            pairs: List of (image_path, label_path) tuples.

        Returns:
            Dictionary mapping class_id to list of pairs containing that class.
        """
        class_pairs: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)
        
        for img_path, label_path in pairs:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_pairs[class_id].append((img_path, label_path))
                        break  # Only use first class for stratification
        
        return class_pairs

    def split_data(
        self,
        pairs: List[Tuple[Path, Path]]
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        Split data into train, validation, and test sets.

        Args:
            pairs: List of (image_path, label_path) tuples.

        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs).
        """
        random.seed(self.random_seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:n_train + n_val]
        test_pairs = shuffled[n_train + n_val:]
        
        logger.info(f"Split sizes - Train: {len(train_pairs)}, "
                   f"Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs

    def stratified_split(
        self,
        pairs: List[Tuple[Path, Path]]
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        Perform stratified split based on class distribution.

        Args:
            pairs: List of (image_path, label_path) tuples.

        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs).
        """
        random.seed(self.random_seed)
        
        class_pairs = self.get_class_distribution(pairs)
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        for class_id, class_data in class_pairs.items():
            random.shuffle(class_data)
            
            n_total = len(class_data)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            train_pairs.extend(class_data[:n_train])
            val_pairs.extend(class_data[n_train:n_train + n_val])
            test_pairs.extend(class_data[n_train + n_val:])
        
        # Remove duplicates while preserving order
        train_pairs = list(dict.fromkeys(train_pairs))
        val_pairs = list(dict.fromkeys(val_pairs))
        test_pairs = list(dict.fromkeys(test_pairs))
        
        logger.info(f"Stratified split sizes - Train: {len(train_pairs)}, "
                   f"Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs

    def copy_files(
        self,
        pairs: List[Tuple[Path, Path]],
        dest_images: Path,
        dest_labels: Path
    ) -> None:
        """
        Copy image and label files to destination directories.

        Args:
            pairs: List of (image_path, label_path) tuples.
            dest_images: Destination directory for images.
            dest_labels: Destination directory for labels.
        """
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        
        for img_path, label_path in pairs:
            shutil.copy2(img_path, dest_images / img_path.name)
            shutil.copy2(label_path, dest_labels / label_path.name)

    def generate_dataset_yaml(self, output_path: Path) -> None:
        """
        Generate YOLO26-compatible dataset.yaml file.

        Args:
            output_path: Path to save the dataset.yaml file.
        """
        splits_path = self.project_root / self.config["data"]["splits_path"]
        
        # Class names from config
        class_names = [c["name"] for c in self.config["classes"]]
        
        dataset_config = {
            "path": str(self.project_root / "data"),
            "train": "splits/train/images",
            "val": "splits/val/images",
            "test": "splits/test/images",
            "nc": len(class_names),
            "names": class_names
        }
        
        with open(output_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Generated dataset.yaml at: {output_path}")

    def verify_no_leakage(
        self,
        train_pairs: List[Tuple[Path, Path]],
        val_pairs: List[Tuple[Path, Path]],
        test_pairs: List[Tuple[Path, Path]]
    ) -> bool:
        """
        Verify there is no data leakage between splits.

        Args:
            train_pairs: Training set pairs.
            val_pairs: Validation set pairs.
            test_pairs: Test set pairs.

        Returns:
            bool: True if no leakage detected.
        """
        train_names = {p[0].stem for p in train_pairs}
        val_names = {p[0].stem for p in val_pairs}
        test_names = {p[0].stem for p in test_pairs}
        
        train_val_overlap = train_names & val_names
        train_test_overlap = train_names & test_names
        val_test_overlap = val_names & test_names
        
        if train_val_overlap:
            logger.error(f"Train-Val overlap detected: {len(train_val_overlap)} files")
            return False
        
        if train_test_overlap:
            logger.error(f"Train-Test overlap detected: {len(train_test_overlap)} files")
            return False
        
        if val_test_overlap:
            logger.error(f"Val-Test overlap detected: {len(val_test_overlap)} files")
            return False
        
        logger.info("No data leakage detected")
        return True

    def log_statistics(
        self,
        train_pairs: List[Tuple[Path, Path]],
        val_pairs: List[Tuple[Path, Path]],
        test_pairs: List[Tuple[Path, Path]]
    ) -> None:
        """
        Log split statistics.

        Args:
            train_pairs: Training set pairs.
            val_pairs: Validation set pairs.
            test_pairs: Test set pairs.
        """
        total = len(train_pairs) + len(val_pairs) + len(test_pairs)
        
        logger.info("=" * 50)
        logger.info("Dataset Split Statistics")
        logger.info("=" * 50)
        logger.info(f"Total samples: {total}")
        logger.info(f"Training:   {len(train_pairs):5d} ({100 * len(train_pairs) / total:.1f}%)")
        logger.info(f"Validation: {len(val_pairs):5d} ({100 * len(val_pairs) / total:.1f}%)")
        logger.info(f"Test:       {len(test_pairs):5d} ({100 * len(test_pairs) / total:.1f}%)")
        logger.info("=" * 50)

    def create_splits(
        self,
        images_dir: Optional[Path] = None,
        labels_dir: Optional[Path] = None
    ) -> None:
        """
        Create train/val/test splits from processed data.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing labels.
        """
        # Default paths
        if images_dir is None:
            images_dir = self.project_root / self.config["data"]["processed_path"] / "images"
        if labels_dir is None:
            labels_dir = self.project_root / self.config["data"]["processed_path"] / "labels"
        
        # Find pairs
        pairs = self.find_image_label_pairs(images_dir, labels_dir)
        
        if len(pairs) == 0:
            logger.error("No image-label pairs found. Please download and organize datasets first.")
            return
        
        # Split data
        if self.stratify:
            train_pairs, val_pairs, test_pairs = self.stratified_split(pairs)
        else:
            train_pairs, val_pairs, test_pairs = self.split_data(pairs)
        
        # Verify no leakage
        self.verify_no_leakage(train_pairs, val_pairs, test_pairs)
        
        # Log statistics
        self.log_statistics(train_pairs, val_pairs, test_pairs)
        
        # Copy files to split directories
        splits_base = self.project_root / self.config["data"]["splits_path"]
        
        logger.info("Copying training files...")
        self.copy_files(
            train_pairs,
            splits_base / "train" / "images",
            splits_base / "train" / "labels"
        )
        
        logger.info("Copying validation files...")
        self.copy_files(
            val_pairs,
            splits_base / "val" / "images",
            splits_base / "val" / "labels"
        )
        
        logger.info("Copying test files...")
        self.copy_files(
            test_pairs,
            splits_base / "test" / "images",
            splits_base / "test" / "labels"
        )
        
        # Generate dataset.yaml
        self.generate_dataset_yaml(self.project_root / "data" / "dataset.yaml")
        
        logger.info("Dataset splitting complete")


def main() -> None:
    """Main entry point for dataset splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to images directory"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Path to labels directory"
    )
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(config_path=args.config)
    
    images_dir = Path(args.images_dir) if args.images_dir else None
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    
    splitter.create_splits(images_dir=images_dir, labels_dir=labels_dir)


if __name__ == "__main__":
    main()
