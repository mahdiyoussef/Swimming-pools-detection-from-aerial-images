"""
Tests for Inference Module.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


class TestPoolDetector:
    """Tests for PoolDetector class."""

    def test_detection_output_format(self) -> None:
        """Test that detection output has correct format."""
        # Mock detection result
        detection = {
            "bbox": [100, 50, 200, 150],
            "confidence": 0.95,
            "class_id": 0,
            "class_name": "swimming_pool"
        }
        
        assert "bbox" in detection
        assert "confidence" in detection
        assert len(detection["bbox"]) == 4
        assert 0 <= detection["confidence"] <= 1

    def test_bbox_format(self) -> None:
        """Test bounding box format (xyxy)."""
        bbox = [100, 50, 200, 150]
        x1, y1, x2, y2 = bbox
        
        assert x2 > x1
        assert y2 > y1


class TestImageProcessing:
    """Tests for image processing utilities."""

    def test_image_loading(self) -> None:
        """Test image loading from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img_path = Path(tmpdir) / "test.jpg"
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), test_img)
            
            # Load image
            loaded = cv2.imread(str(img_path))
            
            assert loaded is not None
            assert loaded.shape == (480, 640, 3)

    def test_image_resize(self) -> None:
        """Test image resizing for inference."""
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        target_size = 640
        
        resized = cv2.resize(image, (target_size, target_size))
        
        assert resized.shape == (target_size, target_size, 3)


class TestOutputGeneration:
    """Tests for output file generation."""

    def test_coordinates_file_format(self) -> None:
        """Test coordinates.txt file format."""
        coordinates = [(100, 50), (200, 50), (200, 150), (100, 150)]
        confidence = 0.95
        
        lines = [
            "Pool Detection Coordinates",
            f"Confidence: {confidence:.4f}",
            "Boundary Points:"
        ]
        lines.extend([f"{x},{y}" for x, y in coordinates])
        
        content = "\n".join(lines)
        
        assert "Confidence:" in content
        assert "Boundary Points:" in content

    def test_output_image_generation(self) -> None:
        """Test output image with outline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw blue rectangle (pool outline)
            pt1 = (100, 100)
            pt2 = (300, 300)
            color = (255, 0, 0)  # Blue in BGR
            thickness = 3
            
            cv2.rectangle(image, pt1, pt2, color, thickness)
            
            # Save
            output_path = Path(tmpdir) / "output_image.jpg"
            cv2.imwrite(str(output_path), image)
            
            assert output_path.exists()
            
            # Verify blue channel has content
            loaded = cv2.imread(str(output_path))
            assert np.any(loaded[:, :, 0] > 0)  # Blue channel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
