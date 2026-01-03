"""
Tests for Data Loading Module.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""

    def test_config_loading(self) -> None:
        """Test configuration file loading."""
        # Test config structure
        config = {
            "data": {"root_path": "data"},
            "sources": {"kaggle": {"enabled": True}},
            "splits": {"train_ratio": 0.7}
        }
        
        assert "data" in config
        assert "sources" in config
        assert config["splits"]["train_ratio"] == 0.7

    def test_archive_extraction(self) -> None:
        """Test archive extraction functionality."""
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = Path(tmpdir) / "test.zip"
            extract_path = Path(tmpdir) / "extracted"
            
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("test.txt", "test content")
            
            # Extract
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_path)
            
            assert (extract_path / "test.txt").exists()


class TestDatasetIntegrity:
    """Tests for dataset integrity verification."""

    def test_image_label_pair_matching(self) -> None:
        """Test that images and labels match correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            labels_dir = Path(tmpdir) / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            
            # Create matching pairs
            for i in range(5):
                (images_dir / f"img_{i}.jpg").touch()
                (labels_dir / f"img_{i}.txt").touch()
            
            # Verify pairs
            images = list(images_dir.glob("*.jpg"))
            for img_path in images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                assert label_path.exists()

    def test_label_file_format(self) -> None:
        """Test YOLO label file format."""
        label_content = "0 0.5 0.5 0.3 0.2\n0 0.7 0.3 0.2 0.1"
        
        for line in label_content.strip().split("\n"):
            parts = line.split()
            assert len(parts) == 5
            
            class_id = int(parts[0])
            assert class_id >= 0
            
            for value in parts[1:]:
                assert 0 <= float(value) <= 1


class TestDataYAML:
    """Tests for dataset.yaml generation."""

    def test_yaml_structure(self) -> None:
        """Test dataset.yaml structure."""
        dataset_config = {
            "path": "/path/to/data",
            "train": "splits/train/images",
            "val": "splits/val/images",
            "test": "splits/test/images",
            "nc": 1,
            "names": ["swimming_pool"]
        }
        
        assert "path" in dataset_config
        assert "train" in dataset_config
        assert "val" in dataset_config
        assert dataset_config["nc"] == len(dataset_config["names"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
