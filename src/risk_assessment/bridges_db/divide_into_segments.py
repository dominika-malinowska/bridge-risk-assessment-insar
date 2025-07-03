"""
This script reads a shapefile with polygons and divides them into segments.
"""

import os
import geopandas as gpd

from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.geometry import GeometryCollection
from shapely.ops import linemerge, unary_union

from utils.find_repo import find_repo_root
from risk_assessment.bridges_db.utils.bridge_divisions import (
    divide_into_segments,
    get_longest_linestring,
    approximate_centerline,
)


if __name__ == "__main__":
    # Find the repo root by looking for a marker file (e.g., .git)
    repo_root = find_repo_root()

    # Define the path to the data
    data_path = os.path.join(repo_root, "data", "bridges_db")

    # Define the output paths
    input_path_shp_poly = os.path.join(data_path, "lsb_OSM_polygons.shp")
    input_path_shp_lines = os.path.join(data_path, "lsb_OSM_lines.shp")

    output_path_shp = os.path.join(data_path, "nbi_segments.shp")

    # Define number of sections to divide the line into
    N_SECTIONS = 5

    # Read the shp file from input_path_shp
    gdf_poly = gpd.read_file(input_path_shp_poly)
    gdf_poly = gdf_poly.set_crs(epsg=4326)  # Set the original CRS to WGS84
    gdf_poly = gdf_poly.to_crs(epsg=3857)  # Convert to Web Mercator

    gdf_lines = gpd.read_file(input_path_shp_lines).set_crs(epsg=4326).to_crs(epsg=3857)

    # TESTING
    # gdf_poly = gdf_poly.iloc[:2]
    # gdf_poly = gdf_poly[gdf_poly['ID'].isin([198, 337, 301])]
    # gdf_poly = gdf_poly[gdf_poly["ID"].isin([22])]

    # Add a new column to the GeoDataFrame to store the centerline
    gdf_poly["segments"] = None

    # Loop over each row in the GeoDataFrame and for each polygon,
    # find the centerline and divide it into segments
    for index, row in gdf_poly.iterrows():
        # print(index)
        polygon = row["geometry"]
        bridge_id = row["ID"]
        # print(bridge_id)

        # If the polygon is a MultiPolygon, merge them into a single Polygon
        if isinstance(polygon, MultiPolygon):
            print(
                f"Multipolygon detected for ID {bridge_id}, trying to merge them with a unary_union"
            )

            polygon = unary_union(polygon)

            if isinstance(polygon, MultiPolygon):
                print(
                    "Multipolygon still exists, trying to merge them with a convex hull"
                )
                polygon = polygon.convex_hull

        # If the polygon is a Polygon, find the centerline and divide it into segments
        if isinstance(polygon, Polygon):
            # Read all lines and select the longest
            lines = gdf_lines[gdf_lines["ID"] == bridge_id].geometry
            # Ensure that the GeoSeries is not empty
            if not lines.empty:
                combined_geometry = lines.unary_union
            else:
                combined_geometry = GeometryCollection()

            # If multilinestring than merge them
            if isinstance(combined_geometry, LineString):
                longest_line = combined_geometry
            else:
                longest_line = get_longest_linestring(linemerge(combined_geometry))

            # If the longest is within 15% length from the bridge lenght, use it as the centerline.
            # Otherwise, try to construct a new one
            # If if doesn't work, use the longest line as the centerline
            if abs(longest_line.length - row["Total Leng"]) < 0.15 * row["Total Leng"]:
                centerline = longest_line
            else:
                # centerline = construct_centerline(polygon, interpolation_distance=1)
                centerline = approximate_centerline(polygon, num_points=30)

            # Added this for 3rd paper to avoid errors
            if centerline is None:
                centerline = longest_line

            # Check if the centerline is None
            # If it is, the bridge needs a different solution and scripts continues to the next bridge
            if centerline is None:
                print(
                    f"bridge with ID {bridge_id} needs a different solutions, bridge length: {row['Total Leng']}"
                )
                continue

            # Divide into segments and make a check to see if the line are roughly equal length
            lines, mean_length, std_dev_length, coeff_var_length = divide_into_segments(
                centerline, N_SECTIONS
            )

            # Update the centerline and segments in the GeoDataFrame
            gdf_poly.at[index, "line"] = centerline
            gdf_poly.at[index, "segments"] = lines
            # print(len(centerline))

        # If the polygon is not a Polygon, print a message
        else:
            print(f"bridge with ID {bridge_id} is not a polygon")

    # Remove the geometry column
    gdf_poly = gdf_poly.drop(columns="geometry")

    # Make the line column the new geometry of the gdf
    gdf_poly = gdf_poly.rename(columns={"line": "geometry"})

    # Convert the 'geometry' column to a GeoSeries
    gdf_poly["geometry"] = gpd.GeoSeries(gdf_poly["geometry"])

    # Convert the DataFrame to a GeoDataFrame
    gdf_poly = gpd.GeoDataFrame(gdf_poly, geometry="geometry")

    # Set the CRS of the GeoDataFrame
    gdf_poly = gdf_poly.set_crs(epsg=3857)  # Set the original CRS to WGS84
    gdf_poly = gdf_poly.to_crs(epsg=4326)  # Convert to Web Mercator

    # Export the GeoDataFrame to a new shapefile
    gdf_poly.drop("segments", axis=1).to_file(
        output_path_shp.replace(".shp", "_centerline.shp")
    )

    # Remove the geometry column
    gdf_poly = gdf_poly.drop(columns="geometry")

    # Loop over each segment and export it to a new shapefile
    for i in range(N_SECTIONS):
        # Create a new DataFrame with the i-th segment of each geometry
        # Create a new DataFrame with all columns from gdf_poly and the i-th segment of each geometry
        df = gdf_poly.copy()
        df["segments"] = df["segments"].apply(
            lambda segments: segments[i] if isinstance(segments, list) else None
        )

        # df = pd.DataFrame(gdf_poly['segments'].apply(lambda segments: segments[i]
        # if isinstance(segments, list) else None))

        # Create a new GeoDataFrame from the DataFrame
        gdf = gpd.GeoDataFrame(df, geometry="segments")

        gdf = gdf.set_crs(epsg=3857)  # Set the original CRS to WGS84
        gdf = gdf.to_crs(epsg=4326)  # Convert to Web Mercator

        # Export the GeoDataFrame to a new shapefile
        segment_output_path_shp = output_path_shp.replace(".shp", f"_segment_{i+1}.shp")
        gdf.to_file(segment_output_path_shp)
