"""
This module contains a function that generates a grid of tiles from
a set of bounding box coordinates that are saved in a GeoDataFrame.
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


def create_tiles(min_lon, min_lat, max_lon, max_lat, step_size=1):
    # Floor and ceil the coordinates to ensure they are integers
    min_lon, min_lat, max_lon, max_lat = (
        math.floor(min_lon),
        math.floor(min_lat),
        math.ceil(max_lon),
        math.ceil(max_lat),
    )

    # Initialize lists to store the coordinates of the tiles
    min_latitudes = []
    max_latitudes = []
    min_longitudes = []
    max_longitudes = []

    # Generate the coordinates of the tiles
    for lat in np.arange(min_lat, max_lat, step_size):
        for lon in np.arange(min_lon, max_lon, step_size):
            min_latitudes.append(lat)
            max_latitudes.append(lat + step_size)
            min_longitudes.append(lon)
            max_longitudes.append(lon + step_size)

    # Create a DataFrame from the coordinates
    tiles_df = pd.DataFrame(
        {
            "minlat": min_latitudes,
            "maxlat": max_latitudes,
            "minlon": min_longitudes,
            "maxlon": max_longitudes,
        }
    )

    # Create a list of Polygon objects from the bounding box coordinates
    polygons = [
        Polygon(
            [
                (row["minlon"], row["minlat"]),
                (row["maxlon"], row["minlat"]),
                (row["maxlon"], row["maxlat"]),
                (row["minlon"], row["maxlat"]),
            ]
        )
        for _, row in tiles_df.iterrows()
    ]

    # Create a GeoDataFrame from the list of polygons
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons)

    # Set the CRS of the GeoDataFrame to WGS84
    polygons_gdf.crs = "EPSG:4326"

    return polygons_gdf
