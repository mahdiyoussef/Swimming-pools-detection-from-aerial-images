"""
Dataset Download Module for Swimming Pool Detection System.

This module provides functionality to download datasets from Kaggle
and Roboflow for training the swimming pool detection model.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Download and organize datasets from Kaggle and Roboflow.

    This class handles the complete workflow of downloading, extracting,
    and organizing datasets for YOLOv11 training.

    Attributes:
        config: Dataset configuration dictionary.
        project_root: Root path of the project.
        raw_path: Path to store raw downloaded data.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the DatasetDownloader.

        Args:
            config_path: Path to the dataset configuration file.
                        If None, uses default config location.
        """
        self.project_root = Path(__file__).resolve().parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "dataset_config.yaml"
        
        self.config = self._load_config(config_path)
        self.raw_path = self.project_root / self.config["data"]["raw_path"]
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            dict: Configuration dictionary.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If configuration file is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config

    def initialize_kaggle_api(self) -> "KaggleApi":
        """
        Initialize and authenticate the Kaggle API.

        Loads API key from .env file and creates kaggle.json if needed.

        Returns:
            KaggleApi: Authenticated Kaggle API instance.

        Raises:
            ImportError: If kaggle package is not installed.
            ValueError: If Kaggle credentials are not configured.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "Kaggle package not installed. Install with: pip install kaggle"
            )
        
        # Load .env file from project parent directory
        env_path = self.project_root.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from: {env_path}")
        
        # Get API key from environment
        api_key = os.environ.get("kaggle_api_key") or os.environ.get("KAGGLE_API_KEY")
        
        # Kaggle looks in different locations depending on OS/version
        # Check both ~/.kaggle and ~/.config/kaggle
        kaggle_dirs = [
            Path.home() / ".kaggle",
            Path.home() / ".config" / "kaggle"
        ]
        
        kaggle_config_exists = any((d / "kaggle.json").exists() for d in kaggle_dirs)
        
        if api_key and not kaggle_config_exists:
            # Create kaggle.json in both locations for compatibility
            if ":" in api_key:
                username, key = api_key.split(":", 1)
            else:
                username = "kaggle_user"
                key = api_key
            
            credentials = {"username": username, "key": key}
            
            for kaggle_dir in kaggle_dirs:
                kaggle_dir.mkdir(parents=True, exist_ok=True)
                kaggle_config = kaggle_dir / "kaggle.json"
                
                with open(kaggle_config, "w") as f:
                    json.dump(credentials, f)
                
                os.chmod(kaggle_config, 0o600)
                logger.info(f"Created Kaggle credentials at: {kaggle_config}")
            kaggle_config_exists = True
        
        if not kaggle_config_exists and not api_key:
            raise ValueError(
                f"Kaggle credentials not found. Either:\n"
                f"1. Set kaggle_api_key in .env file\n"
                f"2. Place kaggle.json in ~/.kaggle or ~/.config/kaggle"
            )
        
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authenticated successfully")
        return api

    def download_from_kaggle(self, dataset_name: Optional[str] = None) -> Path:
        """
        Download dataset from Kaggle.

        Args:
            dataset_name: Kaggle dataset identifier (owner/dataset-name).
                         If None, uses config value.

        Returns:
            Path: Path to the downloaded dataset directory.

        Raises:
            ValueError: If dataset download fails.
        """
        if dataset_name is None:
            dataset_name = self.config["sources"]["kaggle"]["dataset_name"]
        
        api = self.initialize_kaggle_api()
        download_path = self.raw_path / "kaggle"
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"Destination: {download_path}")
        
        try:
            api.dataset_download_files(
                dataset_name,
                path=str(download_path),
                unzip=True
            )
            logger.info("Kaggle dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            raise ValueError(f"Kaggle download failed: {e}")
        
        return download_path

    def download_from_roboflow(
        self,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[int] = None,
        format_type: str = "yolov11"
    ) -> Path:
        """
        Download dataset from Roboflow.

        Args:
            workspace: Roboflow workspace name.
            project: Roboflow project name.
            version: Dataset version number.
            format_type: Export format (default: yolov11).

        Returns:
            Path: Path to the downloaded dataset directory.

        Raises:
            ValueError: If Roboflow API key is not set or download fails.
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "Roboflow package not installed. Install with: pip install roboflow"
            )
        
        # Get API key from environment
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Get your API key from Roboflow settings."
            )
        
        # Use config values if not provided
        if workspace is None:
            workspace = self.config["sources"]["roboflow"]["workspace"]
        if project is None:
            project = self.config["sources"]["roboflow"]["project"]
        if version is None:
            version = self.config["sources"]["roboflow"]["version"]
        
        download_path = self.raw_path / "roboflow"
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading from Roboflow: {workspace}/{project} v{version}")
        
        try:
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download(
                model_format=format_type,
                location=str(download_path)
            )
            logger.info("Roboflow dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download Roboflow dataset: {e}")
            raise ValueError(f"Roboflow download failed: {e}")
        
        return download_path

    def extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """
        Extract a zip archive.

        Args:
            archive_path: Path to the zip file.
            extract_to: Destination directory for extraction.

        Raises:
            FileNotFoundError: If archive doesn't exist.
            zipfile.BadZipFile: If file is not a valid zip archive.
        """
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting: {archive_path}")
        
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.info(f"Extracted to: {extract_to}")

    def verify_integrity(self, dataset_path: Path) -> bool:
        """
        Verify dataset integrity by checking for required files.

        Args:
            dataset_path: Path to the dataset directory.

        Returns:
            bool: True if dataset structure is valid, False otherwise.
        """
        # Check for images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))
            images.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        # Check for labels
        label_files = list(dataset_path.rglob("*.txt"))
        
        logger.info(f"Found {len(images)} images")
        logger.info(f"Found {len(label_files)} label files")
        
        if len(images) == 0:
            logger.warning("No images found in dataset")
            return False
        
        if len(label_files) == 0:
            logger.warning("No label files found in dataset")
            return False
        
        return True

    def organize_files(self, source_path: Path, target_path: Optional[Path] = None) -> None:
        """
        Organize dataset files into a standard structure.

        Args:
            source_path: Path to the source dataset.
            target_path: Target path for organized files. If None, uses processed_path.
        """
        if target_path is None:
            target_path = self.project_root / self.config["data"]["processed_path"]
        
        target_path.mkdir(parents=True, exist_ok=True)
        images_path = target_path / "images"
        labels_path = target_path / "labels"
        images_path.mkdir(exist_ok=True)
        labels_path.mkdir(exist_ok=True)
        
        # Find and copy images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_count = 0
        
        for ext in image_extensions:
            for img_file in source_path.rglob(f"*{ext}"):
                dest = images_path / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    image_count += 1
            for img_file in source_path.rglob(f"*{ext.upper()}"):
                dest = images_path / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    image_count += 1
        
        # Find and copy labels
        label_count = 0
        for label_file in source_path.rglob("*.txt"):
            # Skip non-annotation files
            if label_file.name in ["classes.txt", "notes.txt", "README.txt"]:
                continue
            dest = labels_path / label_file.name
            if not dest.exists():
                shutil.copy2(label_file, dest)
                label_count += 1
        
        logger.info(f"Organized {image_count} images and {label_count} labels")

    def download_all(self) -> None:
        """
        Download datasets from all enabled sources.
        """
        logger.info("Starting dataset download...")
        
        # Download from Kaggle if enabled
        if self.config["sources"]["kaggle"]["enabled"]:
            try:
                kaggle_path = self.download_from_kaggle()
                if self.verify_integrity(kaggle_path):
                    self.organize_files(kaggle_path)
                    logger.info("Kaggle dataset processed successfully")
            except Exception as e:
                logger.error(f"Kaggle download failed: {e}")
        
        # Download from Roboflow if enabled
        if self.config["sources"]["roboflow"]["enabled"]:
            try:
                roboflow_path = self.download_from_roboflow()
                if self.verify_integrity(roboflow_path):
                    self.organize_files(roboflow_path)
                    logger.info("Roboflow dataset processed successfully")
            except Exception as e:
                logger.error(f"Roboflow download failed: {e}")
        
        logger.info("Dataset download complete")


def main() -> None:
    """Main entry point for dataset download."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download datasets for swimming pool detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["kaggle", "roboflow", "all"],
        default="all",
        help="Dataset source to download from"
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=None,
        help="Override Kaggle dataset name"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(config_path=args.config)
    
    if args.source == "kaggle":
        path = downloader.download_from_kaggle(dataset_name=args.kaggle_dataset)
        if downloader.verify_integrity(path):
            downloader.organize_files(path)
    elif args.source == "roboflow":
        path = downloader.download_from_roboflow()
        if downloader.verify_integrity(path):
            downloader.organize_files(path)
    else:
        downloader.download_all()


if __name__ == "__main__":
    main()
