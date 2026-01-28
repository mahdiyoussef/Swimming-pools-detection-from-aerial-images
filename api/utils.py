import math
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from typing import Tuple, List

def deg2num(lat_deg, lon_deg, zoom):
    """Convert Lat/Lng to tile X/Y."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Convert tile X/Y to Lat/Lng of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_map_frame(
    north: float, 
    south: float, 
    east: float, 
    west: float, 
    zoom: int,
    zoom_boost: int = 2
) -> Tuple[np.ndarray, float, float, float]:
    """
    Fetch and stitch map tiles for a given bounding box at high resolution.
    
    Args:
        north, south, east, west: Bounding box coordinates
        zoom: Base zoom level from the map view
        zoom_boost: Extra zoom levels for higher resolution (default 2)
                   zoom_boost=0: Original resolution
                   zoom_boost=1: 2x resolution (4x pixels)
                   zoom_boost=2: 4x resolution (16x pixels)
    
    Returns: (Stitched Image, Top Latitude, Left Longitude, Meters-Per-Pixel)
    """
    # Apply zoom boost for higher resolution
    effective_zoom = min(zoom + zoom_boost, 20)  # Google max zoom is 20-21
    
    # 1. Get tile range at the effective (boosted) zoom level
    x_min, y_min = deg2num(north, west, effective_zoom)
    x_max, y_max = deg2num(south, east, effective_zoom)
    
    # 2. Setup tile grid
    tile_size = 256
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    
    # Cap size to prevent memory issues (max ~4096x4096 for high-res)
    max_tiles = 256  # 16x16 grid = 4096x4096 pixels max
    if cols * rows > max_tiles:
        # Reduce zoom if too many tiles
        return fetch_map_frame(north, south, east, west, zoom, zoom_boost - 1)

    print(f"[HighRes] Fetching {cols}x{rows} tiles at zoom {effective_zoom} (boost +{zoom_boost})")
    
    canvas = Image.new('RGB', (cols * tile_size, rows * tile_size))
    
    # Google Satellite URL (high quality satellite imagery)
    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    
    # 3. Fetch tiles and stitch
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = url_template.format(x=x, y=y, z=effective_zoom)
            try:
                response = requests.get(url, timeout=10)
                tile = Image.open(BytesIO(response.content))
                canvas.paste(tile, ((x - x_min) * tile_size, (y - y_min) * tile_size))
            except Exception as e:
                print(f"Error fetching tile {x},{y}: {e}")
                
    # 4. Calculate actual top-left of the stitched image
    origin_lat, origin_lng = num2deg(x_min, y_min, effective_zoom)
    
    # 5. Calculate Ground Resolution (m/px) at the EFFECTIVE zoom level
    lat_mid = (north + south) / 2
    mpp = math.cos(math.radians(lat_mid)) * 40075016.686 / (256 * 2**effective_zoom)
    
    print(f"[HighRes] Resolution: {mpp:.4f} m/px, Image size: {cols * tile_size}x{rows * tile_size}")
    
    # 6. Convert to OpenCV format (BGR)
    cv_image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    
    return cv_image, origin_lat, origin_lng, mpp

