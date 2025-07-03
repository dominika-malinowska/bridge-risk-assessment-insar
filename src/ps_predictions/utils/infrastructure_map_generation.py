"""
This script contains a function that can be used to generate
the infrastructure map for a given location.
"""

import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping
from geocube.api.core import make_geocube
from functools import partial

from geocube.rasterize import rasterize_image

from ps_predictions.utils.preprocessing_OSM_data import (
    BuildRoadRailHandler,
    convert_osm_osmconvert,
    map_value,
)


def generate_infrastructure_map(
    lat_f,
    lon_f,
    min_x,
    min_y,
    max_x,
    max_y,
    path_output,
    clipped_osm_pbf_path,
    infrastructure_map_path,
    osm_convert_path,
):
    """
    This function generates the infrastructure map for a given location.

    Arguments:
        lat_f (str): The latitude of the location.
        lon_f (str): The longitude of the location.
        min_x (float): The minimum x-coordinate of the bounding box.
        min_y (float): The minimum y-coordinate of the bounding box.
        max_x (float): The maximum x-coordinate of the bounding box.
        max_y (float): The maximum y-coordinate of the bounding box.
        path_output (str): The directory where the output is stored.
        clipped_osm_pbf_path (str): The path to the clipped OSM PBF file.
        infrastructure_map_path (str): The path to the infrastructure map raster file.
        osm_convert_path (str): The path to the osmconvert executable.
    """

    clipped_o5m_path = os.path.join(
        path_output,
        f"{lat_f}{lon_f}",
        f"{lat_f}{lon_f}.o5m",
    )

    # Check if the clipped O5M file already exists
    if os.path.exists(clipped_o5m_path):
        print(f"Clipped o5m for {lat_f}{lon_f} already exists.", flush=True)
    # If the clipped O5M file does not exist, start the clipping process
    else:
        # Convert the osm pbf to o5m
        convert_osm_osmconvert(osm_convert_path, clipped_osm_pbf_path, clipped_o5m_path)

        # Check if conversion of o5m is done and print the result
        if os.path.exists(clipped_o5m_path):
            print(
                "Converting of o5m for {}{} is done".format(lat_f, lon_f),
                flush=True,
            )
        else:
            print(
                "\n Converting of o5m for {}{} failed. Moving to the next file. \n".format(
                    lat_f, lon_f
                ),
                flush=True,
            )

    #  Record the start time for generating a df
    start_time = time.time()

    # Create a handler for building road and rail data
    handler = BuildRoadRailHandler()

    # Start processing the data file
    handler.apply_file(clipped_o5m_path, locations=True, idx="flex_mem")

    # Create dataframes for buildings, roads, and railways
    df_buil = pd.DataFrame(handler.buildings)
    df_road = pd.DataFrame(handler.roads)
    df_rail = pd.DataFrame(handler.railways)

    # Print the lengths of the dataframes
    print(len(df_buil), len(df_road), len(df_rail), flush=True)

    # Assign types to the dataframes
    df_rail["type"] = "rail"
    df_buil["type"] = "building"
    df_road.rename(columns={"highway": "type"}, inplace=True)

    # Concatenate the dataframes
    frames = [df_rail, df_road, df_buil]
    df_all = pd.concat(frames)

    # Delete the original dataframes to save memory
    del df_rail, df_road, df_buil

    # Record the end time for generating a df and calculate the total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time for df creation: {total_time:.2f} seconds", flush=True)

    # Record start time for raster generation
    start_time_line2 = time.time()

    # Apply the function to create the new 'value' column
    df_all["value"] = df_all.apply(map_value, axis=1)

    # Sort the dataframe in ascending order of 'value'
    # so that when rasterised, the highest value is retained
    # (as geocube by default replaces a value that is already in a pixel)
    df_all.sort_values(by=["value"], inplace=True)

    # Convert the dataframe to a GeoDataFrame
    gdf_OSM = gpd.GeoDataFrame(df_all)

    # Define the bounding box
    bbox = (min_x, min_y, max_x, max_y)
    geom = mapping(box(*bbox))

    # Create a geocube from the GeoDataFrame
    cube = make_geocube(
        gdf_OSM,
        measurements=["value"],
        # resolution - a tuple of the spatial resolution of the returned data (Y, X).
        # This includes the direction (as indicated by a positive or negative number)
        resolution=(-1 / 1200, 1 / 1200),
        geom=geom,
        fill=0,
        rasterize_function=partial(rasterize_image, all_touched=True),
    )

    # Create the directory for infrastructure map if it doesn't exist
    infrmap_dir = os.path.join(path_output, "{}{}".format(lat_f, lon_f), "infr_map")
    if not os.path.exists(infrmap_dir):
        os.makedirs(infrmap_dir)

    # Save the geocube as a raster
    cube.value.rio.to_raster(infrastructure_map_path)

    # Print the time taken for raster generation
    end_time_line2 = time.time()
    time_line2 = end_time_line2 - start_time_line2
    print(f"Time for raster generation: {time_line2:.2f} seconds", flush=True)
