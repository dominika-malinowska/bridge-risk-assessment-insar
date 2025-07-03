"""
This script processes the LSB Database and retrieves the OSM lines and polygons for the bridges.
"""

import os

import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.find_repo import find_repo_root

from risk_assessment.risk_calculation.structural_vulnerability_assessment import (
    get_vulnerability,
)
from risk_assessment.bridges_db.retrieve_osm_lines_for_brdgs import (
    find_OSM_lines_and_polygons,
)
from risk_assessment.risk_calculation.exposure_assessment import (
    get_exposure,
)

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    # Find the repo root by looking for a marker file (e.g., .git)
    repo_root = find_repo_root()

    # Define the path to the data
    data_path = os.path.join(repo_root, "data", "bridges_db")
    lsb_path = os.path.join(data_path, "LSB Database_corrected.csv")

    # Define the output paths
    output_path = os.path.join(data_path, "lsb_OSM_polygons.shp")
    output_path_lines = os.path.join(data_path, "lsb_OSM_lines.shp")
    output_path_lines_csv = os.path.join(data_path, "lsb_OSM_lines.csv")

    # Read the bridges db into a DataFrame
    df = pd.read_csv(lsb_path)

    # Get the vulnerability of the structures
    df = get_vulnerability(df)

    # Convert the DataFrame to a GeoDataFrame by creating Point geometries from Longitude and Latitude
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Drop rows where ID is nan
    gdf = gdf.dropna(subset=["ID"])

    # # Save the GeoDataFrame to a shapefile
    # output_path_gdf = os.path.join(data_path, "LSB Database.shp")
    # gdf.to_file(output_path_gdf)

    # Store the initial length of the GeoDataFrame
    initial_len_df = len(gdf)

    # Decrease size of the gdf for testing
    # gdf = gdf.iloc[:1]
    # gdf = gdf[gdf["ID"].isin([536])]

    # Find the OSM lines and polygons for the bridges
    gdf_lines, gdf_polygons = find_OSM_lines_and_polygons(gdf)

    # Print the number of processed bridges and check if all bridges have been processed
    print(len(gdf_lines), initial_len_df)
    if len(gdf_lines) == initial_len_df:
        print("All bridges have been processed")
    else:
        print("Some bridges have not been processed")

    # Get the exposure of the polygons and lines
    gdf_polygons = get_exposure(gdf_polygons)
    gdf_lines = get_exposure(gdf_lines)

    # Save the GeoDataFrames to file
    gdf_polygons.to_file(output_path)
    gdf_lines.to_file(output_path_lines)
    gdf_lines.to_csv(output_path_lines_csv)
