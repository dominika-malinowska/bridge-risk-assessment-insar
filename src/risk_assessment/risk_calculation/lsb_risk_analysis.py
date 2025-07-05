"""
This script reads the data from the LSB database and the zonal statistics files
and calculates the risk for each bridge based on the exposure, vulnerability, and
# the maximum value of the landslide and subsidence rasters.
# The risk is calculated as the cube root of the product of the maximum value of
# the raster and the normalised exposure and vulnerability values.

"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.find_repo import find_repo_root
from risk_assessment.risk_calculation.utils.monitoring_class_calculations import (
    select_value,
    assign_monitoring_class,
    monitoring_class_stats,
    calculate_geometric_mean,
    determine_snt_availability,
    calculate_risk,
)

if __name__ == "__main__":

    # Set the max value of the monitoring coefficient
    MAX_MONITORING_COEFFICIENT = 1.35

    # Define continents
    continents = [
        "europe",
        "australia",
        "africa",
        "centralamerica",
        "southamerica",
        "northamerica",
        "asia",
    ]

    # Find the repo root by looking for a marker file (e.g., .git)
    repo_root = find_repo_root()

    # Define the path to the data
    data_path = os.path.join(repo_root, "data")

    # Define the paths to the specific files
    bridges_db_path = os.path.join(
        data_path, "bridges_db", "LSB Database_corrected.csv"
    )
    exposure_path = os.path.join(data_path, "bridges_db", "lsb_OSM_lines.csv")
    subsidence_path = os.path.join(data_path, "zonal_stats", "bridges_subsidence.csv")
    landslides_path = os.path.join(data_path, "zonal_stats", "bridges_landslides.csv")

    # Define path for storing the results
    results_path = os.path.join(data_path, "risk_assessment")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Define the columns to read
    columns_to_read = [
        "ID",
        "Latitude",
        "Longitude",
        "Global Region",
        "Country/Region",
        "Crossing",
        "Type",
        "Total Length",
        "Monitoring",
        "Construction Finished",
        "Material: Cable/Truss",
        "Material: Deck",
        "Material: Piers/Pylons",
        "Maximum Span",
        "Health",
    ]

    # Read the shapefile into a GeoDataFrame with specific columns
    bridges_gdf = gpd.read_file(bridges_db_path, usecols=columns_to_read)

    # Rename the columns
    bridges_gdf.rename(
        columns={
            "Global Region": "Region",
            "Country/Region": "Country",
            "Total Length": "Length",
            # "Vulnerabil": "Vulnerability",
            "Construction Finished": "Construction year",
            # "Material_": "Material: Cable/Truss",
            # "Material_1": "Material: Deck",
            # "Material_2": "Material: Piers/Pylons",
            # "Maximum Sp": "Maximum Span",
        },
        inplace=True,
    )

    # Remove rows with missing or empty 'ID' values
    bridges_gdf = bridges_gdf.dropna(subset=["ID"])
    bridges_gdf = bridges_gdf[bridges_gdf["ID"].astype(str).str.strip() != ""]

    # Convert 'TRUE'/'FALSE' strings to actual booleans
    bridges_gdf["Monitoring"] = bridges_gdf["Monitoring"].replace(
        {"TRUE": True, "FALSE": False}
    )

    # Create point geometries from latitude and longitude
    bridges_gdf["geometry"] = bridges_gdf.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )

    # Assign data types to the columns
    bridges_gdf = bridges_gdf.astype(
        {
            "ID": "int64",
            "Region": "category",
            "Country": "category",
            "Crossing": "category",
            "Type": "category",
            "Length": "float64",
            "Monitoring": "int",
            "Construction year": "int64",
            "Material: Cable/Truss": "category",
            "Material: Deck": "category",
            "Material: Piers/Pylons": "category",
            "Maximum Span": "float64",
            "Health": "category",
        }
    )

    # Define the columns to load from each CSV file
    columns_to_load_exposure = ["ID", "Exposure", "Vulnerability_class"]
    columns_to_load_subsidence = [
        "ID",
        "su_count",
        "su_min",
        "su_mean",
        "su_max",
        "su_std",
        "su_sum",
        "su_nodata",
    ]
    columns_to_load_landslides = [
        "ID",
        "ls_count",
        "ls_min",
        "ls_mean",
        "ls_max",
        "ls_std",
        "ls_sum",
        "ls_nodata",
    ]

    # Read CSV files with exposure and hazards into a DataFrame, select specific columns,
    # and merge with the GeoDataFrame
    for path, cols in [
        (exposure_path, columns_to_load_exposure),
        (subsidence_path, columns_to_load_subsidence),
        (landslides_path, columns_to_load_landslides),
    ]:
        df = pd.read_csv(path, usecols=cols)
        bridges_gdf = bridges_gdf.merge(df, on="ID")

    # Rename the columns
    bridges_gdf.rename(
        columns={
            "Vulnerability_class": "Vulnerability",
        },
        inplace=True,
    )

    # Read PS density and SNT availability from continental files and merge with main gdf
    for continent in continents:
        for segment_id in [1, 2, 3, 4, 5]:

            path = os.path.join(
                data_path,
                "Zonal_stats",
                "bridges_PSdens_{}_{}.csv".format(continent, segment_id),
            )

            cols = [
                "ID",
                "ps_count_{}_{}".format(continent, segment_id),
                "ps_min_{}_{}".format(continent, segment_id),
                "ps_mean_{}_{}".format(continent, segment_id),
                "ps_max_{}_{}".format(continent, segment_id),
                "ps_std_{}_{}".format(continent, segment_id),
                "ps_sum_{}_{}".format(continent, segment_id),
                "ps_nodata_{}_{}".format(continent, segment_id),
                "ps_zero_count_{}_{}".format(continent, segment_id),
                "asc_{}_1sc_time".format(segment_id),
                "dsc_{}_1sc_time".format(segment_id),
                "asc_{}_2sc_time".format(segment_id),
                "dsc_{}_2sc_time".format(segment_id),
                "length_{}".format(segment_id),
                "azimuth_{}".format(segment_id),
            ]

            df = pd.read_csv(path, usecols=cols)

            # Columns with Sentinel availability are repeated for each continent
            # so take only the one from Europe and ignore the rest
            if continent != "europe":
                columns_to_drop = [
                    "asc_{}_1sc_time".format(segment_id),
                    "dsc_{}_1sc_time".format(segment_id),
                    "asc_{}_2sc_time".format(segment_id),
                    "dsc_{}_2sc_time".format(segment_id),
                    "length_{}".format(segment_id),
                    "azimuth_{}".format(segment_id),
                ]

                df = df.drop(columns=columns_to_drop)

            bridges_gdf = bridges_gdf.merge(df, on="ID")

    # # Write the GeoDataFrame to a new CSV file
    # lsb_gdf.to_csv(os.path.join(data_path, 'Vulnerability', 'temp.csv'))

    # Normalise exposure and vulnerability
    bridges_gdf["Exposure_norm"] = bridges_gdf["Exposure"] / 5
    bridges_gdf["Vulnerability_norm"] = bridges_gdf["Vulnerability"] / 5

    bridges_gdf["ls_max"] = bridges_gdf["ls_max"].fillna(0)
    bridges_gdf["su_max"] = bridges_gdf["su_max"].fillna(0)

    # Calculate the multi-hazard
    bridges_gdf["Multi-hazard"] = bridges_gdf.apply(
        lambda row: calculate_geometric_mean(row, ["su_max", "ls_max"]),
        axis=1,
    )

    # Calculate risk without monitoring
    bridges_gdf["Landslide_risk_without_monitoring"] = bridges_gdf.apply(
        lambda row: calculate_geometric_mean(
            row, ["ls_max", "Exposure_norm", "Vulnerability_norm"]
        ),
        axis=1,
    )
    bridges_gdf["Subsidence_risk_without_monitoring"] = bridges_gdf.apply(
        lambda row: calculate_geometric_mean(
            row, ["su_max", "Exposure_norm", "Vulnerability_norm"]
        ),
        axis=1,
    )

    bridges_gdf["Multi-hazard_risk_without_monitoring"] = bridges_gdf.apply(
        lambda row: calculate_geometric_mean(
            row, ["Multi-hazard", "Exposure_norm", "Vulnerability_norm"]
        ),
        axis=1,
    )

    # print(vulnerability_lsb_gdf.head())

    # PS was processed per continent so that results are spread between few columns.
    # Create one column that lists PS mean for each bridge.
    # If a bridge was processed twice because it is covered by two continental datasets
    # then the max value will be taken.
    for i in range(1, 6):
        bridges_gdf["ps_mean_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "mean", continents), axis=1
        )
        bridges_gdf["ps_count_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "count", continents), axis=1
        )
        bridges_gdf["ps_zero_count_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "zero_count", continents), axis=1
        )
        bridges_gdf["ps_min_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "min", continents), axis=1
        )
        bridges_gdf["ps_max_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "max", continents), axis=1
        )
        bridges_gdf["ps_std_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "std", continents), axis=1
        )
        bridges_gdf["ps_sum_world_{}".format(i)] = bridges_gdf.apply(
            lambda row: select_value(row, i, "sum", continents), axis=1
        )

    # Count how many pixels a bridge covers as a sum of pixels covered by each segment individually
    cols_to_sum = [
        "ps_count_world_1",
        "ps_count_world_2",
        "ps_count_world_3",
        "ps_count_world_4",
        "ps_count_world_5",
    ]
    bridges_gdf["ps_count"] = bridges_gdf[cols_to_sum].sum(axis=1)

    # Count number of pixels over a bridge with no PS
    cols_to_sum_zero = [
        "ps_zero_count_world_1",
        "ps_zero_count_world_2",
        "ps_zero_count_world_3",
        "ps_zero_count_world_4",
        "ps_zero_count_world_5",
    ]

    # Clean columns that have '--' values and convert them to be Nans
    # These are bridges not covered by PS data
    bridges_gdf[cols_to_sum_zero] = (
        bridges_gdf[cols_to_sum_zero]
        .replace("--", np.nan)
        .astype("float")
        .astype("Int64")
    )
    bridges_gdf["ps_zero_count"] = bridges_gdf[cols_to_sum_zero].sum(axis=1)

    # Calculate percentage of PS availability for each bridge
    # as a share of pixels with PS data over all pixels
    # Set 0 availbility if no data for a bridge
    bridges_gdf["ps_avail_perc"] = np.where(
        (bridges_gdf["ps_zero_count"] == 0) & (bridges_gdf["ps_count"] == 0),
        0,  # when no data for a bridge (both values 0)
        1 - bridges_gdf["ps_zero_count"] / bridges_gdf["ps_count"],
    )

    # Calculate percentage of PS availability for each segment
    for i in range(1, 6):
        bridges_gdf[f"ps_avail_perc_{i}"] = np.where(
            (bridges_gdf[f"ps_zero_count_world_{i}"].fillna(0) == 0)
            & (bridges_gdf[f"ps_count_world_{i}"] == 0),
            0,  # when no data for a bridge (both values 0)
            1
            - bridges_gdf[f"ps_zero_count_world_{i}"]
            / bridges_gdf[f"ps_count_world_{i}"],
        )

    # Create columns with mean, std, and Coefficient of Variation (CV) across segments
    cols_to_mean = [
        "ps_mean_world_1",
        "ps_mean_world_2",
        "ps_mean_world_3",
        "ps_mean_world_4",
        "ps_mean_world_5",
    ]
    bridges_gdf["ps_mean_brdg"] = bridges_gdf[cols_to_mean].mean(axis=1)
    bridges_gdf["ps_std_brdg"] = bridges_gdf[cols_to_mean].std(axis=1)
    bridges_gdf["ps_cv_brdg"] = (
        bridges_gdf["ps_std_brdg"] / bridges_gdf["ps_mean_brdg"] * 100
    )

    # Create a column with count of PSs per 100 meters for each segment
    for i in range(1, 6):
        bridges_gdf[f"ps_count_100m_{i}"] = (
            bridges_gdf[f"ps_sum_world_{i}"] * 100 / bridges_gdf[f"length_{i}"]
        )

    # Get the maximum time between acquisitions for ascending and descending orbits
    bridges_gdf["asc_1sc_time"] = bridges_gdf[
        [
            "asc_1_1sc_time",
            "asc_2_1sc_time",
            "asc_3_1sc_time",
            "asc_4_1sc_time",
            "asc_5_1sc_time",
        ]
    ].max(axis=1)

    bridges_gdf["dsc_1sc_time"] = bridges_gdf[
        [
            "dsc_1_1sc_time",
            "dsc_2_1sc_time",
            "dsc_3_1sc_time",
            "dsc_4_1sc_time",
            "dsc_5_1sc_time",
        ]
    ].max(axis=1)

    bridges_gdf["asc_2sc_time"] = bridges_gdf[
        [
            "asc_1_2sc_time",
            "asc_2_2sc_time",
            "asc_3_2sc_time",
            "asc_4_2sc_time",
            "asc_5_2sc_time",
        ]
    ].max(axis=1)

    bridges_gdf["dsc_2sc_time"] = bridges_gdf[
        [
            "dsc_1_2sc_time",
            "dsc_2_2sc_time",
            "dsc_3_2sc_time",
            "dsc_4_2sc_time",
            "dsc_5_2sc_time",
        ]
    ].max(axis=1)

    bridges_gdf["SNT_availability_1sc"] = bridges_gdf.apply(
        lambda row: determine_snt_availability(row, snt_status="_1sc"), axis=1
    )
    bridges_gdf["SNT_availability_2sc"] = bridges_gdf.apply(
        lambda row: determine_snt_availability(row, snt_status="_2sc"), axis=1
    )

    # Assign monitoring class based on PS and SNT availability
    bridges_gdf = assign_monitoring_class(bridges_gdf, "_1sc")
    bridges_gdf = assign_monitoring_class(bridges_gdf, "_2sc")

    # Calculate statistics for monitoring class
    # and assign final class based on weighted average of the class of each segment
    bridges_gdf = monitoring_class_stats(
        bridges_gdf, "_1sc", MAX_MONITORING_COEFFICIENT
    )
    bridges_gdf = monitoring_class_stats(
        bridges_gdf, "_2sc", MAX_MONITORING_COEFFICIENT
    )

    # Correct vulnerability with availability of monitoring
    # If SHM is available, the vulnerability is multiplied by 1.35
    # If spaceborne monitoring is available, the vulnerability is multiplied
    # by the monitoring class calculated based on PS availability and scaled
    # to be between 1 (no monitoring) and 1.35 (great SNT monitoring)
    bridges_gdf["Monitoring_scaled_shm"] = bridges_gdf["Monitoring"].apply(
        lambda x: MAX_MONITORING_COEFFICIENT if x == 0 else 1
    )
    bridges_gdf["Vulnerability_norm_monitoring_shm"] = (
        bridges_gdf["Vulnerability_norm"] * bridges_gdf["Monitoring_scaled_shm"]
    ).clip(upper=1)

    bridges_gdf["Vulnerability_norm_monitoring_1sc"] = (
        bridges_gdf["Vulnerability_norm"] * bridges_gdf["Monitoring_scaled_1sc"]
    ).clip(upper=1)

    bridges_gdf["Vulnerability_norm_monitoring_2sc"] = (
        bridges_gdf["Vulnerability_norm"] * bridges_gdf["Monitoring_scaled_2sc"]
    ).clip(upper=1)

    # Create a column that stores the combined value of shm and space monitoring
    # Take the minimum of the two values (as smaller means better monitoring)
    bridges_gdf["Monitoring_shm_and_2sc"] = bridges_gdf[
        ["Monitoring_scaled_shm", "Monitoring_scaled_2sc"]
    ].min(axis=1)

    # Correct vulnerability with availability of monitoring
    bridges_gdf["Vulnerability_norm_monitoring_com"] = (
        bridges_gdf["Vulnerability_norm"] * bridges_gdf["Monitoring_shm_and_2sc"]
    ).clip(upper=1)

    # Calculate updated risk with monitoring included
    calculate_risk(
        bridges_gdf,
        "Landslide",
        "ls_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_shm",
    )
    calculate_risk(
        bridges_gdf,
        "Subsidence",
        "su_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_shm",
    )
    calculate_risk(
        bridges_gdf,
        "Landslide",
        "ls_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_1sc",
    )
    calculate_risk(
        bridges_gdf,
        "Subsidence",
        "su_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_1sc",
    )
    calculate_risk(
        bridges_gdf,
        "Landslide",
        "ls_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_2sc",
    )
    calculate_risk(
        bridges_gdf,
        "Subsidence",
        "su_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_2sc",
    )

    calculate_risk(
        bridges_gdf,
        "Landslide",
        "ls_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_com",
    )
    calculate_risk(
        bridges_gdf,
        "Subsidence",
        "su_max",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_com",
    )

    # Multi-hazard risk calculation with monitoring
    calculate_risk(
        bridges_gdf,
        "Multi-hazard",
        "Multi-hazard",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_shm",
    )
    calculate_risk(
        bridges_gdf,
        "Multi-hazard",
        "Multi-hazard",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_1sc",
    )
    calculate_risk(
        bridges_gdf,
        "Multi-hazard",
        "Multi-hazard",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_2sc",
    )
    calculate_risk(
        bridges_gdf,
        "Multi-hazard",
        "Multi-hazard",
        "Exposure_norm",
        "Vulnerability_norm_monitoring_com",
    )

    # Remove all columns with 'continent' in name
    for continent in continents:
        bridges_gdf = bridges_gdf.loc[
            :, ~bridges_gdf.columns.str.contains("{}".format(continent))
        ]

    # # Write the GeoDataFrame to a new CSV file
    # bridges_gdf.to_csv(
    #     os.path.join(
    #         data_path,
    #         results_path,
    #         f"lsb_risk_analysis_{MAX_MONITORING_COEFFICIENT}.csv",
    #     ),
    #     index=False,
    # )

    # # Write the GeoDataFrame to a pickle
    # bridges_gdf.to_pickle(
    #     os.path.join(
    #         data_path,
    #         results_path,
    #         f"lsb_risk_analysis_{MAX_MONITORING_COEFFICIENT}.pkl",
    #     )
    # )

    # List of columns to keep
    columns_to_keep = [
        "ID",
        "geometry",
        "Region",
        "Country",
        "Crossing",
        "Type",
        "Length",
        "azimuth_3",
        "Construction year",
        "Material: Cable/Truss",
        "Material: Deck",
        "Material: Piers/Pylons",
        "Maximum Span",
        "Health",
        "su_max",
        "ls_max",
        "Exposure_norm",
        "Vulnerability_norm",
        "Multi-hazard",
        "Landslide_risk_without_monitoring",
        "Subsidence_risk_without_monitoring",
        "Multi-hazard_risk_without_monitoring",
        "SNT_availability_1sc",
        "SNT_availability_2sc",
        "Monitoring",
        "Monitoring_scaled_shm",
        "Monitoring_PS_availability_only",
        "Monitoring_1_PS_availability_only",
        "Monitoring_2_PS_availability_only",
        "Monitoring_3_PS_availability_only",
        "Monitoring_4_PS_availability_only",
        "Monitoring_5_PS_availability_only",
        "Monitoring_1sc",
        "Monitoring_scaled_1sc",
        "Monitoring_2sc",
        "Monitoring_scaled_2sc",
        "Monitoring_shm_and_2sc",
        "Vulnerability_norm_monitoring_shm",
        "Vulnerability_norm_monitoring_1sc",
        "Vulnerability_norm_monitoring_2sc",
        "Vulnerability_norm_monitoring_com",
        "Landslide_risk_shm",
        "Landslide_risk_1sc",
        "Landslide_risk_2sc",
        "Landslide_risk_com",
        "Subsidence_risk_shm",
        "Subsidence_risk_1sc",
        "Subsidence_risk_2sc",
        "Subsidence_risk_com",
        "Multi-hazard_risk_shm",
        "Multi-hazard_risk_1sc",
        "Multi-hazard_risk_2sc",
        "Multi-hazard_risk_com",
    ]

    # Filter the GeoDataFrame to keep only the specified columns
    filtered_bridges_gdf = bridges_gdf[columns_to_keep]

    # Write the filtered GeoDataFrame to a new CSV file
    filtered_bridges_gdf.to_csv(
        os.path.join(
            data_path,
            results_path,
            f"lsb_risk_analysis_filtered_{MAX_MONITORING_COEFFICIENT}.csv",
        ),
        index=False,
    )

    # Write the filtered GeoDataFrame to a new pickle file
    filtered_bridges_gdf.to_pickle(
        os.path.join(
            data_path,
            results_path,
            f"lsb_risk_analysis_filtered_{MAX_MONITORING_COEFFICIENT}.pkl",
        )
    )

    # # Write the GeoDataFrame to a shapefile
    # columns_to_keep = [
    #     "ID",
    #     "geometry",
    #     "Monitoring_PS_availability_only",
    #     "Monitoring_1_PS_availability_only",
    #     "Monitoring_2_PS_availability_only",
    #     "Monitoring_3_PS_availability_only",
    #     "Monitoring_4_PS_availability_only",
    #     "Monitoring_5_PS_availability_only",
    #     "Monitoring_2sc",
    # ]

    # filtered_bridges_gdf = filtered_bridges_gdf[columns_to_keep]

    # rename_dict = {
    #     "Monitoring_PS_availability_only": "M_PS_only",
    #     "Monitoring_1_PS_availability_only": "M_1_PS_only",
    #     "Monitoring_2_PS_availability_only": "M_2_PS_only",
    #     "Monitoring_3_PS_availability_only": "M_3_PS_only",
    #     "Monitoring_4_PS_availability_only": "M_4_PS_only",
    #     "Monitoring_5_PS_availability_only": "M_5_PS_only",
    #     "Monitoring_2sc": "M_2sc",
    # }

    # # Rename the columns
    # filtered_bridges_gdf = filtered_bridges_gdf.rename(columns=rename_dict)

    # filtered_bridges_gdf.to_file(
    #     os.path.join(
    #         data_path,
    #         results_path,
    #         f"lsb_risk_analysis_{MAX_MONITORING_COEFFICIENT}.shp",
    #     )
    # )
