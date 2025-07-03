import os
import math
import requests
from pathlib import Path


# Define functions #
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    This function calculates the haversine distance between two points on the Earth's surface.

    Arguments:
        lat1 (float): Latitude of the first point in degrees
        lon1 (float): Longitude of the first point in degrees
        lat2 (float): Latitude of the second point in degrees
        lon2 (float): Longitude of the second point in degrees

    Returns:
        float: The haversine distance between the two points in kilometers
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance


def calculate_area(x, y):
    """
    This function calculates the area of a rectangle defined by two latitude and longitude points.

    Arguments:
        x (float): Latitude of the first point in degrees
        y (float): Longitude of the first point in degrees

    Returns:
        float: The area of the rectangle in square kilometers
    """

    # Calculate distances
    lat_distance = haversine_distance(y[0], x[0], y[1], x[0])
    lon_distance = haversine_distance(y[0], x[0], y[0], x[1])

    # Calculate area in square kilometers
    area_km2 = lat_distance * lon_distance

    return area_km2


def download_osm(filename=None, clipped_osm_pbf=None):
    url = f"https://download.geofabrik.de/{filename}"

    # Check if file already exists
    out_path = clipped_osm_pbf if clipped_osm_pbf is not None else filename
    if os.path.exists(out_path):
        print(f"File already exists at {out_path}, skipping download.")
        return

    print(f"Downloading {filename}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    # Ensure the parent directory exists
    if clipped_osm_pbf is not None:
        Path(clipped_osm_pbf).parent.mkdir(parents=True, exist_ok=True)
        out_path = clipped_osm_pbf
    else:
        out_path = filename

    with open(out_path, "wb") as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    print(f"\nDownload complete! File saved as {out_path}")
