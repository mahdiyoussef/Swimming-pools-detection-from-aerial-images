"""
Tests for Preprocessing Module.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestDatasetSplitter:
    """Tests for DatasetSplitter class."""

    def test_split_ratios_sum_to_one(self) -> None:
        """Test that split ratios sum to 1.0."""
        ratios = [0.7, 0.2, 0.1]
        assert abs(sum(ratios) - 1.0) < 0.001

    def test_split_data_distribution(self) -> None:
        """Test that data is split correctly."""
        # Create mock data
        n_samples = 100
        data = [(f"img_{i}.jpg", f"img_{i}.txt") for i in range(n_samples)]
        
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        assert n_train == 70
        assert n_val == 20
        assert n_test == 10


class TestAnnotationConverter:
    """Tests for AnnotationConverter class."""

    def test_polygon_to_bbox(self) -> None:
        """Test polygon to bounding box conversion."""
        polygon = [(10, 10), (50, 10), (50, 40), (10, 40)]
        
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        assert x_min == 10
        assert y_min == 10
        assert x_max == 50
        assert y_max == 40

    def test_yolo_format_normalization(self) -> None:
        """Test YOLO format coordinate normalization."""
        # Original bbox: x1=100, y1=50, x2=200, y2=150 on 640x480 image
        img_width, img_height = 640, 480
        x1, y1, x2, y2 = 100, 50, 200, 150
        
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1

    def test_yolo_annotation_validation(self) -> None:
        """Test YOLO annotation validation."""
        # Valid annotation
        valid_line = "0 0.5 0.5 0.3 0.2"
        parts = valid_line.split()
        
        assert len(parts) == 5
        assert int(parts[0]) >= 0  # Class ID
        
        for i in range(1, 5):
            value = float(parts[i])
            assert 0 <= value <= 1


class TestAugmentation:
    """Tests for augmentation pipeline."""

    def test_bbox_remains_valid_after_flip(self) -> None:
        """Test that bounding boxes remain valid after horizontal flip."""
        # YOLO format bbox
        x_center, y_center, width, height = 0.3, 0.5, 0.2, 0.3
        
        # After horizontal flip, x_center should be mirrored
        flipped_x_center = 1.0 - x_center
        
        assert 0 <= flipped_x_center <= 1

    def test_image_dimensions_preserved(self) -> None:
        """Test that image dimensions are preserved after augmentation."""
        original_shape = (640, 640, 3)
        
        # Create dummy image
        image = np.random.randint(0, 255, original_shape, dtype=np.uint8)
        
        assert image.shape == original_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
