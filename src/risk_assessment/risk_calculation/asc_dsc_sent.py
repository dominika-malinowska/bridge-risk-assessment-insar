"""
The code reads all geometries from shp, filters it so that only those 
acquired with IW mode are left, and then sorts them in ascending 
and descending order based on their geometry
"""

import geopandas as gpd
import math
from shapely.geometry import Polygon, LineString


def calculate_angle(linestring):
    """
    This function calculates the angle of a linestring

    Arguments:
        linestring (LineString): A linestring geometry

    Returns:
        float: The angle of the linestring in degrees
    """
    point1, point2 = linestring.coords[:2]
    angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    return math.degrees(angle)


def identify_asc_and_dsc(file_path, output_file):
    """
    Process a single shapefile to identify ascending and descending geometries.

    Parameters:
    file_path (str): The path to the shapefile.
    output_file (str): The path to save the processed shapefile.

    Returns:
    gpd.GeoDataFrame: The processed GeoDataFrame.
    """
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(file_path)

    # Leave only those acquisitions that were acquired with IW mode
    gdf = gdf[gdf["Mode"] == "IW"]

    # Calculate angle of each linestring to identify ascending and descending geometries
    gdf["angle"] = gdf["geometry"].apply(calculate_angle)

    # Add a column to the gdf that indicates whether the flight is ascending or descending
    gdf["flight_dir"] = gdf["angle"].apply(
        lambda x: "ascending" if x > 0 else "descending"
    )

    # Convert 3D linestrings to 2D
    gdf["geometry"] = gdf["geometry"].apply(
        lambda line: LineString([(x, y) for x, y, z in line.coords])
    )

    # Convert linestring geometries to polygons
    gdf["geometry"] = gdf["geometry"].apply(lambda x: Polygon(x))

    # Save gdf to a shp
    gdf.to_file(output_file)

    return gdf
