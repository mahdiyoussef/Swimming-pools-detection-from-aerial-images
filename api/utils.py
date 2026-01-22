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

def fetch_map_frame(north: float, south: float, east: float, west: float, zoom: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Fetch and stitch map tiles for a given bounding box.
    Returns: (Stitched Image, Top Latitude, Left Longitude, Meters-Per-Pixel)
    """
    # 1. Get tile range
    x_min, y_min = deg2num(north, west, zoom)
    x_max, y_max = deg2num(south, east, zoom)
    
    # 2. Setup tile grid
    tile_size = 256
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    
    # Cap size to prevent memory issues (approx 2048x2048 max)
    if cols * rows > 100:
        zoom -= 1
        return fetch_map_frame(north, south, east, west, zoom)

    canvas = Image.new('RGB', (cols * tile_size, rows * tile_size))
    
    # Google Satellite URL (Hybrid style)
    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    
    # 3. Fetch tiles and stitch
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = url_template.format(x=x, y=y, z=zoom)
            try:
                response = requests.get(url, timeout=5)
                tile = Image.open(BytesIO(response.content))
                canvas.paste(tile, ((x - x_min) * tile_size, (y - y_min) * tile_size))
            except Exception as e:
                print(f"Error fetching tile {x},{y}: {e}")
                
    # 4. Calculate actual top-left of the stitched image
    origin_lat, origin_lng = num2deg(x_min, y_min, zoom)
    
    # 5. Calculate Ground Resolution (m/px)
    # Ref: meters_per_pixel = cos(lat) * 2 * pi * 6378137 / (256 * 2^zoom)
    lat_mid = (north + south) / 2
    mpp = math.cos(math.radians(lat_mid)) * 40075016.686 / (256 * 2**zoom)
    
    # 6. Convert to OpenCV format (BGR)
    cv_image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    
    return cv_image, origin_lat, origin_lng, mpp
