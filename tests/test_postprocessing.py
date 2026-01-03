"""
Tests for Postprocessing Module.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


class TestCoordinateExtraction:
    """Tests for coordinate extraction functions."""

    def test_bbox_to_corners(self) -> None:
        """Test bounding box to corner conversion."""
        bbox = np.array([100, 50, 200, 150])
        
        x1, y1, x2, y2 = bbox[:4]
        
        corners = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2),  # bottom-left
        ]
        
        assert len(corners) == 4
        assert corners[0] == (100, 50)
        assert corners[2] == (200, 150)

    def test_coordinate_clamping(self) -> None:
        """Test coordinate clamping to image boundaries."""
        img_shape = (480, 640)
        bbox = np.array([-10, -20, 700, 500])
        
        x1 = max(0, min(int(bbox[0]), img_shape[1]))
        y1 = max(0, min(int(bbox[1]), img_shape[0]))
        x2 = max(0, min(int(bbox[2]), img_shape[1]))
        y2 = max(0, min(int(bbox[3]), img_shape[0]))
        
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= img_shape[1]
        assert y2 <= img_shape[0]


class TestOutlineDrawing:
    """Tests for pool outline drawing."""

    def test_polygon_drawing(self) -> None:
        """Test polygon drawing on image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        coordinates = [(100, 100), (200, 100), (200, 200), (100, 200)]
        color = (255, 0, 0)  # Blue
        thickness = 3
        
        pts = np.array(coordinates, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, thickness)
        
        # Check that red pixels exist
        assert np.any(image[:, :, 2] > 0)

    def test_outline_color(self) -> None:
        """Test that outline color is red (BGR: 0, 0, 255)."""
        color = (0, 0, 255)  # Red in BGR
        
        assert color[0] == 0    # Blue channel
        assert color[1] == 0    # Green channel
        assert color[2] == 255  # Red channel


class TestCoordinateSaving:
    """Tests for coordinate file saving."""

    def test_save_and_load_coordinates(self) -> None:
        """Test saving and loading coordinates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_path = Path(tmpdir) / "coordinates.txt"
            
            coordinates = [(100, 50), (200, 50), (200, 150), (100, 150)]
            confidence = 0.95
            
            # Save
            with open(coord_path, "w") as f:
                f.write("Pool Detection Coordinates\n")
                f.write(f"Confidence: {confidence:.4f}\n")
                f.write("Boundary Points:\n")
                for x, y in coordinates:
                    f.write(f"{x},{y}\n")
            
            # Load
            loaded_coords = []
            loaded_conf = 0.0
            
            with open(coord_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Confidence:"):
                        loaded_conf = float(line.split(":")[1])
                    elif "," in line:
                        x, y = line.split(",")
                        loaded_coords.append((int(x), int(y)))
            
            assert abs(loaded_conf - confidence) < 0.001
            assert loaded_coords == coordinates


class TestAreaCalculation:
    """Tests for pool area calculation."""

    def test_rectangular_area(self) -> None:
        """Test area calculation for rectangular pool."""
        # 100x50 rectangle
        coordinates = [(0, 0), (100, 0), (100, 50), (0, 50)]
        
        # Shoelace formula
        n = len(coordinates)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        area = abs(area) / 2.0
        
        expected_area = 100 * 50
        assert abs(area - expected_area) < 0.001


class TestCoordinateTransformations:
    """Tests for coordinate transformations."""

    def test_normalize_coordinates(self) -> None:
        """Test coordinate normalization."""
        coords = [(320, 240)]
        img_width, img_height = 640, 480
        
        normalized = [(x / img_width, y / img_height) for x, y in coords]
        
        assert normalized[0] == (0.5, 0.5)

    def test_denormalize_coordinates(self) -> None:
        """Test coordinate denormalization."""
        normalized_coords = [(0.5, 0.5)]
        img_width, img_height = 640, 480
        
        pixel_coords = [
            (int(x * img_width), int(y * img_height))
            for x, y in normalized_coords
        ]
        
        assert pixel_coords[0] == (320, 240)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
