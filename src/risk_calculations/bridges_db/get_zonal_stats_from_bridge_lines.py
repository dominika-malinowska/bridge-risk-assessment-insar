"""
This script computes zonal statistics on PS density per segment
and on the hazard level for landslides and subsidence per bridge
for the bridges in the input shapefile.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

from ps_predictions.second_paper.risk_calculation.asc_dsc_sent import (
    identify_asc_and_dsc,
)


# Define custom function for counting all pixels with no PSs
def count_zeros(x):
    """
    This function counts the number of zeros in an array.

    Arguments:
        x (numpy.ndarray): The input array.

    Returns:
        int: The number of zeros in the input array.
    """
    return np.sum(x == 0)


def check_coverage(df, pixel_values_df, segment_id, prefix):
    """
    The function checks if bridges is covered by the acquisition
    and adds information about the coverage to the pixel_values_df.

    Arguments:
        df (GeoDataFrame): The GeoDataFrame with the bridges.
        pixel_values_df (GeoDataFrame): The GeoDataFrame with the pixel values.
        segment_id (int): The segment ID.
        prefix (str): The prefix for the column name.

    Returns:
        None
    """
    column_name = f"{prefix}_{segment_id}"
    pixel_values_df[column_name] = pixel_values_df["geometry"].apply(
        lambda x: 1 if df["geometry"].contains(x).any() else 0
    )


def find_overlapping_geometries(df, pixel_values_df):
    """
    The function finds the overlapping geometries between the bridges
    and the acquisition.

    Arguments:
        df (GeoDataFrame): The GeoDataFrame with the bridges.
        pixel_values_df (GeoDataFrame): The GeoDataFrame with the pixel values.

    Returns:
        GeoDataFrame: The GeoDataFrame with the overlapping geometries.
    """
    return gpd.sjoin(df, pixel_values_df, how="inner", op="intersects")


def calculate_azimuth(line):
    """
    Calculate the azimuth (bearing) in degrees between the start and end points of a LineString.
    """
    if line.is_empty or line is None:
        return np.nan

    start_point = line.coords[0]
    end_point = line.coords[-1]

    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]

    azimuth = np.degrees(np.arctan2(delta_x, delta_y))
    azimuth = (azimuth + 360) % 360  # Normalize to 0-360 degrees

    return azimuth


if __name__ == "__main__":
    # Define the data path
    data_path = os.path.join("/mnt/g/RISK_PAPER")

    # Read shp file with Sentinel availability and generate information about ascending and descending acquisitions
    snt_1A_2023_path = os.path.join(
        "/mnt/g/RISK_PAPER/Vulnerability/Sentinel_availability/S1A.shp"
    )

    snt_aval_path_1sc = (
        "/mnt/g/RISK_PAPER/Vulnerability/Sentinel_availability/asc_dsc_sent.shp"
    )

    if os.path.exists(snt_aval_path_1sc):
        # If the file exists, read the GeoDataFrame from the file
        gdf_snt_aval_1sc = gpd.read_file(snt_aval_path_1sc)
    else:
        # If the file does not exist, generate the GeoDataFrame
        gdf_snt_aval_1sc = identify_asc_and_dsc(snt_1A_2023_path, snt_aval_path_1sc)

    # Convert timestamps to datetime format and remove hours, minutes, and seconds
    gdf_snt_aval_1sc["Name"] = pd.to_datetime(gdf_snt_aval_1sc["Name"]).dt.floor("D")

    # Create separate asc and dsc dataframes
    gdf_snt_aval_asc_1sc = gdf_snt_aval_1sc[
        gdf_snt_aval_1sc["flight_dir"] == "ascending"
    ]
    gdf_snt_aval_dsc_1sc = gdf_snt_aval_1sc[
        gdf_snt_aval_1sc["flight_dir"] == "descending"
    ]

    # Read shp file with Sentinel availability pre-failure of SNT-1B
    snt_1AB_2021_path = os.path.join(
        "/mnt/g/RISK_PAPER/Vulnerability/Sentinel_availability_prefailure/SNT-1AB-20210402-20210604.shp"
    )

    snt_aval_path_2sc = "/mnt/g/RISK_PAPER/Vulnerability/Sentinel_availability_prefailure/asc_dsc_sent.shp"

    if os.path.exists(snt_aval_path_2sc):
        # If the file exists, read the GeoDataFrame from the file
        gdf_snt_aval_2sc = gpd.read_file(snt_aval_path_2sc)
    else:
        # If the file does not exist, generate the GeoDataFrame
        gdf_snt_aval_2sc = identify_asc_and_dsc(snt_1AB_2021_path, snt_aval_path_2sc)

    # Convert timestamps to datetime format and remove hours, minutes, and seconds
    gdf_snt_aval_2sc["Name"] = pd.to_datetime(gdf_snt_aval_2sc["Name"]).dt.floor("D")

    # Create separate asc and dsc dataframes
    gdf_snt_aval_asc_2sc = gdf_snt_aval_2sc[
        gdf_snt_aval_2sc["flight_dir"] == "ascending"
    ]
    gdf_snt_aval_dsc_2sc = gdf_snt_aval_2sc[
        gdf_snt_aval_2sc["flight_dir"] == "descending"
    ]

    # Pattern for the path to the input raster with PS density for a continent
    input_path_raster_pattern = os.path.join(
        data_path,
        "Final_predictions_for_paper",
        "merged_tiffs",
        "ml_model_data_{}",
        "merged_{}.tif",
    )

    # Path to the shapefile with central lines
    input_path_shp = os.path.join(
        data_path, "lsb_bridges_database", "new_centerlines_test.shp"
    )

    # Pattern to the shapefile with central lines segmented
    input_path_shp_segment_pattern = input_path_shp.replace(".shp", "_segment_{}.shp")

    # Pattern for the the output CSV file with zonal statistics for PS density per continent and segment
    output_csv_pattern = os.path.join(
        data_path, "Zonal_stats", "bridges_PSdens_{}_{}.csv"
    )

    # Path to input raster with landslides
    input_path_raster_landslide = os.path.join(
        data_path, "Hazard", "Landslides", "LS_TH_normalised.tif"
    )

    # Path to the output CSV file with zonal statistics for landslides
    output_csv_landslides = os.path.join(
        data_path, "Zonal_stats", "bridges_landslides.csv"
    )

    # Path to input raster with subsidence
    input_path_raster_subsidence = os.path.join(
        data_path,
        "Hazard",
        "Differential_ground_movement",
        "natural_and_water",
        "GSH_2040_normalised.tif",
    )

    # Path to the output CSV file with zonal statistics for subsidence
    output_csv_subsidence = os.path.join(
        data_path, "Zonal_stats", "bridges_subsidence.csv"
    )

    # Zonal statistic for PS density separately for each continent and segment #
    for continent in [
        "europe",
        "australia",
        "africa",
        "centralamerica",
        "southamerica",
        "northamerica",
        "asia",
    ]:
        # Replace the continent in the general pattern
        input_path_raster = input_path_raster_pattern.format(continent, continent)

        for segment_id in [1, 2, 3, 4, 5]:
            # Replace the segment ID in the general pattern
            input_path_segment_shp = input_path_shp_segment_pattern.format(segment_id)

            # Remove rows that are missing geometries
            gdf = gpd.read_file(input_path_segment_shp)
            # filter out empty or null geometries
            gdf = gdf[~(gdf["geometry"].is_empty | gdf["geometry"].isna())]

            # Add info about segments length
            gdf = gdf.to_crs(epsg=3857)
            gdf[f"length_{segment_id}"] = gdf["geometry"].length
            gdf = gdf.to_crs(epsg=4326)

            # Calculate azimuth
            gdf[f"azimuth_{segment_id}"] = gdf["geometry"].apply(calculate_azimuth)

            # Compute zonal statistics
            stats = zonal_stats(
                gdf,
                input_path_raster,
                nodata=255,
                all_touched=True,
                geojson_out=True,
                stats="count min mean max std sum nodata",
                add_stats={"zero_count": count_zeros},
            )

            # Convert the statistics to a GeoDataFrame
            pixel_values_df = gpd.GeoDataFrame.from_features(stats).set_crs(epsg=4326)

            # Rename the columns
            pixel_values_df.rename(
                columns={
                    "count": f"ps_count_{continent}_{segment_id}",
                    "min": f"ps_min_{continent}_{segment_id}",
                    "mean": f"ps_mean_{continent}_{segment_id}",
                    "max": f"ps_max_{continent}_{segment_id}",
                    "std": f"ps_std_{continent}_{segment_id}",
                    "sum": f"ps_sum_{continent}_{segment_id}",
                    "nodata": f"ps_nodata_{continent}_{segment_id}",
                    "zero_count": f"ps_zero_count_{continent}_{segment_id}",
                },
                inplace=True,
            )

            # List of tuples containing the DataFrame, prefix, and segment_id
            data_frames = [
                (gdf_snt_aval_asc_1sc, "asc", f"{segment_id}_1sc"),
                (gdf_snt_aval_dsc_1sc, "dsc", f"{segment_id}_1sc"),
                (gdf_snt_aval_asc_2sc, "asc", f"{segment_id}_2sc"),
                (gdf_snt_aval_dsc_2sc, "dsc", f"{segment_id}_2sc"),
            ]

            # Check if bridge segment is covered by asc/dsc
            for df, prefix, seg_id in data_frames:
                check_coverage(df, pixel_values_df, seg_id, prefix)

            # Find time difference between asc and dsc acquisitions for each bridge
            # Find overlapping geometries for each DataFrame
            data_frames = [
                (gdf_snt_aval_1sc, "asc_dsc", f"{segment_id}_1sc"),
                (gdf_snt_aval_2sc, "asc_dsc", f"{segment_id}_2sc"),
            ]
            overlapping_geometries = {
                f"{prefix}_{seg_id}": find_overlapping_geometries(df, pixel_values_df)
                for df, prefix, seg_id in data_frames
            }

            # Access the overlapping geometries as needed
            overlapping_geometries_asc_dsc_1sc = overlapping_geometries[
                f"asc_dsc_{segment_id}_1sc"
            ]
            overlapping_geometries_asc_dsc_2sc = overlapping_geometries[
                f"asc_dsc_{segment_id}_2sc"
            ]

            # For each bridge, find the shortest time between acquisitions
            for index, row in pixel_values_df.iterrows():

                # List of acquisitions DataFrames to check
                acquisitions_list = [
                    (
                        overlapping_geometries_asc_dsc_1sc,
                        f"asc_dsc_{segment_id}_1sc_time",
                    ),
                    (
                        overlapping_geometries_asc_dsc_2sc,
                        f"asc_dsc_{segment_id}_2sc_time",
                    ),
                ]

                for acquisitions_df, time_col in acquisitions_list:
                    # Find all acquisitions that cover the bridge
                    acquisitions = acquisitions_df[acquisitions_df["ID"] == row["ID"]]

                    if not acquisitions.empty:

                        # Sort, filter, and drop duplicates
                        sorted_acquisitions = acquisitions.sort_values(
                            by="Name"
                        ).drop_duplicates(subset=["Name", "flight_dir", "OrbitRelat"])

                        min_diffs = (
                            sorted_acquisitions.groupby(["OrbitRelat", "flight_dir"])[
                                "Name"
                            ]
                            .apply(
                                lambda x: x.diff()
                                .abs()
                                .dt.days.dropna()
                                .astype(int)
                                .min()
                            )
                            .reset_index(name="MinDateDiff")
                        )

                        # get the ascending repeat time
                        pixel_values_df.at[index, time_col.replace("dsc_", "")] = (
                            min_diffs[min_diffs["flight_dir"] == "ascending"][
                                "MinDateDiff"
                            ].min()
                        )

                        pixel_values_df.at[index, time_col.replace("asc_", "")] = (
                            min_diffs[min_diffs["flight_dir"] == "descending"][
                                "MinDateDiff"
                            ].min()
                        )
                        asc_time_temp = pixel_values_df.loc[
                            index, time_col.replace("dsc_", "")
                        ]
                        dsc_time_temp = pixel_values_df.loc[
                            index, time_col.replace("asc_", "")
                        ]

            # Save the GeoDataFrame to a CSV file
            pixel_values_df.to_csv(output_csv_pattern.format(continent, segment_id))

    # Zonal statistics for landslides and differential ground movement #

    # Read the input shapefile with whole bridges (not segmented)
    gdf = gpd.read_file(input_path_shp)
    # filter out empty or null geometries
    gdf = gdf[~(gdf["geometry"].is_empty | gdf["geometry"].isna())]

    # Get zonal statistics for landslide hazard
    stats = zonal_stats(
        gdf,
        input_path_raster_landslide,
        nodata=0,
        all_touched=True,
        geojson_out=True,
        stats="count min mean max std sum nodata",
    )
    # Convert the statistics to a GeoDataFrame and save to csv
    pixel_values_df = gpd.GeoDataFrame.from_features(stats)
    pixel_values_df.rename(
        columns={
            "count": "ls_count",
            "min": "ls_min",
            "mean": "ls_mean",
            "max": "ls_max",
            "std": "ls_std",
            "sum": "ls_sum",
            "nodata": "ls_nodata",
        },
        inplace=True,
    )
    pixel_values_df.to_csv(output_csv_landslides)

    # Repeat the process for differential ground movement
    stats = zonal_stats(
        gdf,
        input_path_raster_subsidence,
        nodata=0,
        all_touched=True,
        geojson_out=True,
        stats="count min mean max std sum nodata",
    )
    pixel_values_df = gpd.GeoDataFrame.from_features(stats)
    pixel_values_df.rename(
        columns={
            "count": "su_count",
            "min": "su_min",
            "mean": "su_mean",
            "max": "su_max",
            "std": "su_std",
            "sum": "su_sum",
            "nodata": "su_nodata",
        },
        inplace=True,
    )
    pixel_values_df.to_csv(output_csv_subsidence)
