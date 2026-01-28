from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.detect_pools import PoolDetector
from inference.postprocessing import (
    extract_pool_contour,
    classify_pool_shape,
    calculate_pool_area,
    calculate_real_world_area,
    pixel_to_latlng
)
from api.utils import fetch_map_frame

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoolAPI")

app = FastAPI(title="Swimming Pool Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR / "weights" / "best.pt"

# Global detector instance
detector = None

class DetectionRequest(BaseModel):
    north: float
    south: float
    east: float
    west: float
    zoom: int

@app.on_event("startup")
async def startup_event():
    global detector
    logger.info(f"Loading model from {MODEL_PATH}...")
    if MODEL_PATH.exists():
        detector = PoolDetector(model_path=str(MODEL_PATH), conf_threshold=0.60)
        logger.info("Model loaded successfully.")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")

@app.get("/api/detections")
async def list_detections() -> List[str]:
    if not OUTPUT_DIR.exists():
        return []
    detections = []
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir() and (item / "detections.geojson").exists():
                detections.append(item.name)
    return sorted(detections)

@app.get("/api/detections/{image_name}/geojson")
async def get_geojson(image_name: str) -> Dict:
    """Read and return the GeoJSON for a specific detection."""
    geojson_path = OUTPUT_DIR / image_name / "detections.geojson"
    
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON not found for this detection")
        
    try:
        with open(geojson_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading GeoJSON: {str(e)}")

@app.post("/api/detect_live")
async def detect_live(req: DetectionRequest):
    """
    Capture a high-resolution map frame and run sliding window detection.
    
    Uses:
    - High-resolution tile fetching (zoom boosted by +2 levels)
    - Sliding window (tiled) detection for accuracy on large images
    - Non-Maximum Suppression to merge overlapping detections
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Fetch HIGH-RESOLUTION map image (zoom_boost=2 for 4x resolution)
        logger.info(f"Fetching high-res tiles for bounds: N={req.north:.4f}, S={req.south:.4f}, E={req.east:.4f}, W={req.west:.4f}")
        image, origin_lat, origin_lng, mpp = fetch_map_frame(
            req.north, req.south, req.east, req.west, req.zoom,
            zoom_boost=2  # Boost zoom by 2 levels for higher resolution
        )
        
        img_h, img_w = image.shape[:2]
        logger.info(f"Fetched image: {img_w}x{img_h} pixels, resolution: {mpp:.4f} m/px")
        
        # 2. Run SLIDING WINDOW (Tiled) Inference
        # This is critical for large high-res images where pools are small relative to image size
        _, raw_detections = detector.detect_tiled(
            image,
            tile_size=512,    # Process 512x512 tiles
            overlap=128,      # 25% overlap for better boundary detection
            image_size=640    # YOLO inference size
        )
        
        logger.info(f"Sliding window detection found {len(raw_detections)} raw detections")
        
        # 3. Process to GeoJSON
        features = []
        for i, det in enumerate(raw_detections):
            bbox = det["bbox"]
            conf = float(det["confidence"])
            
            # Extract high-precision contour
            polygon = extract_pool_contour(image, np.array(bbox))
            shape = classify_pool_shape(polygon)
            
            # Convert to geospatial
            geo_poly = pixel_to_latlng(polygon, origin_lat, origin_lng, mpp)
            
            # Calculate real area
            pix_area = calculate_pool_area(polygon)
            real_area = calculate_real_world_area(pix_area, mpp)
            
            # GeoJSON coordinates [Lng, Lat]
            coordinates = [[lng, lat] for lat, lng in geo_poly]
            if coordinates:
                coordinates.append(coordinates[0]) # Close
                
            features.append({
                "type": "Feature",
                "id": f"live_{i}",
                "properties": {
                    "confidence": conf,
                    "shape": shape,
                    "area_m2": round(real_area, 2),
                    "is_live": True
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                }
            })
        
        logger.info(f"Returning {len(features)} pool features")
            
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "mpp": mpp,
                "zoom": req.zoom,
                "effective_zoom": req.zoom + 2,
                "image_size": f"{img_w}x{img_h}",
                "count": len(features)
            }
        }
        
    except Exception as e:
        logger.error(f"Live detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {
        "status": "healthy" if detector else "loading",
        "model": "YOLOv11s",
        "output_dir": str(OUTPUT_DIR)
    }

if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
