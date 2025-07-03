"""
This script contains functions for dividing bridges into segments.
"""

import pandas as pd
import geopandas as gpd

import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from shapely.ops import substring

import numpy as np
import pygeos


def calculate_angle(line1, line2):
    """
    This function calculates the angle between two lines.

    Arguments:
        line1 (LineString): The first line
        line2 (LineString): The second line

    Returns:
        angle (float): The angle between the two lines in degrees
    """

    # Calculate the direction of the lines
    direction1 = np.array(line1.coords[-1]) - np.array(line1.coords[0])
    direction2 = np.array(line2.coords[-1]) - np.array(line2.coords[0])

    # Normalize the directions
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)

    # Calculate the angle between the lines
    angle = np.arccos(np.clip(np.dot(direction1, direction2), -1.0, 1.0))

    # Convert the angle to degrees
    angle = np.degrees(angle)

    return angle


def close_gaps(df, tolerance):
    """
    This function closes gaps in LineString geometry where it should be contiguous.
    Snaps both lines to a centroid of a gap in between.

    Arguments:
        df (GeoDataFrame): The GeoDataFrame with LineString geometries
        tolerance (float): The tolerance distance for snapping the lines

    Returns:
        gdf (GeoDataFrame): The GeoDataFrame with closed gaps
    """
    geom = df.geometry.to_numpy()
    geom = pygeos.from_shapely(geom)
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    buffered = pygeos.buffer(points, tolerance)

    dissolved = pygeos.union_all(buffered)

    exploded = [
        pygeos.get_geometry(dissolved, i)
        for i in range(pygeos.get_num_geometries(dissolved))
    ]

    centroids = pygeos.centroid(exploded)

    snapped = pygeos.snap(geom, pygeos.union_all(centroids), tolerance)

    # Assuming 'snapped' is your array of PyGEOS Geometry objects
    geometries = [pygeos.to_shapely(geom) for geom in snapped]

    # Create a new GeoDataFrame with the geometries
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(range(len(geometries)), columns=["id"]), geometry=geometries
    )

    return gdf


def get_two_longest_linestrings(multilinestring):
    """
    This function returns the two longest LineStrings from a MultiLineString.

    Arguments:
        multilinestring (MultiLineString): The MultiLineString to extract the two longest LineStrings from

    Returns:
        MultiLineString: The two longest LineStrings
    """

    if isinstance(multilinestring, MultiLineString):
        sorted_lines = sorted(
            multilinestring.geoms, key=lambda line: line.length, reverse=True
        )
        return MultiLineString(sorted_lines[:2])
    else:
        return MultiLineString([multilinestring])


def approximate_centerline(input_geometry, num_points=100):

    # Buffer and simplify the input geometry to avoid issues with the Voronoi diagram
    input_geometry = input_geometry.buffer(2, cap_style="flat")
    input_geometry = input_geometry.simplify(5)
    input_geometry = input_geometry.buffer(-50)

    # Get the bounds of the polygon
    minx, miny, maxx, maxy = input_geometry.bounds

    # If the distance between the min and max x coordinates is less than the distance between
    # the min and max y coordinates,
    # create a series of horizontal lines
    if maxx - minx < maxy - miny:
        # Create a series of horizontal lines
        y_coords = np.linspace(miny, maxy, num_points)
        horizontal_lines = [LineString([(minx, y), (maxx, y)]) for y in y_coords]

        # Intersect each line with the polygon
        intersections = [line.intersection(input_geometry) for line in horizontal_lines]

    # else, create a series of vertical lines
    else:
        x_coords = np.linspace(minx, maxx, num_points)
        vertical_lines = [LineString([(x, miny), (x, maxy)]) for x in x_coords]

        # Intersect each line with the polygon
        intersections = [line.intersection(input_geometry) for line in vertical_lines]

    # For each intersection, find the midpoint
    midpoints = []
    for intersection in intersections:
        if not intersection.is_empty:
            if intersection.geom_type == "MultiLineString":
                for line in intersection.geoms:
                    midpoints.append(line.centroid)
            else:
                midpoints.append(intersection.centroid)

    # Create a line from the midpoints
    if len(midpoints) > 1:
        centerline = LineString(midpoints)
        # Smooth the centerline
        smoothed_centerline = centerline.simplify(5)
        return smoothed_centerline
    else:
        return None


def construct_centerline(
    input_geometry,
    interpolation_distance=0.5,
    min_length_ratio=0.05,
    take_longest=False,
    longest_count=1,
):
    """
    This function constructs the centerline of a polygon.

    Arguments:
        input_geometry (Polygon): The input polygon
        interpolation_distance (float): The distance between the Voronoi vertices
        min_length_ratio (float): The minimum length ratio of the centerline to the longest line
        take_longest (bool): If True, the longest line is taken as the centerline
        longest_count (int): The number of longest lines to take as the centerline

    Returns:
        LineString: The centerline of the polygon
    """

    # Buffer and simplify the input geometry to avoid issues with the Voronoi diagram
    input_geometry = input_geometry.buffer(2, cap_style="flat")
    input_geometry = input_geometry.simplify(5)
    input_geometry = input_geometry.buffer(-50)

    # source: https://github.com/fitodic/centerline/issues/42
    # find the voronoi verticies (equivalent to Centerline._get_voronoi_vertices_and_ridges())
    # The smaller the interpolation_distance, the more points will be added to the geometry,
    # and the resulting Voronoi diagram will be more detailed.
    # However, this also increases the computational complexity. So, it's a trade-off between
    # precision and performance.

    # To have smaler verticies (equivalent to Centerline._get_densified_borders())
    borders = input_geometry.segmentize(interpolation_distance)

    # equivalent to the scipy.spatial.Voronoi
    voronoied = shapely.voronoi_polygons(
        borders, extend_to=input_geometry, only_edges=True
    )

    # to select only the linestring within the input geometry
    # (equivalent to Centerline._linestring_is_within_input_geometry)
    centerlines = gpd.sjoin(
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms)),
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(input_geometry)),
        predicate="within",
    )

    # If the centerlines are empty, return None
    if centerlines.empty:
        print(
            "The input polygon is really small which might cause errors. Decreasing interpolation distance might help."
        )
        centerlines = gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms))

        return None

    # Merge the centerlines
    multi_line = centerlines.unary_union
    multi_line = linemerge(multi_line)

    # If the centerline is a LineString, return it
    # Otherwise, merge the LineStrings
    if multi_line.geom_type == "LineString":
        return multi_line
    else:

        # Identify the main line (assuming it's the longest one)
        main_line = max(multi_line.geoms, key=lambda line: line.length)

        # Filter the lines based on the minimum length ratio
        filtered_lines = [
            line
            for line in multi_line.geoms
            if line.length >= min_length_ratio * main_line.length
        ]
        merged_line = MultiLineString(filtered_lines)
        merged_line = linemerge(merged_line)

        merged_line = merged_line.simplify(5)

        if merged_line.geom_type == "LineString":
            return merged_line

        else:
            # Assuming 'merged_line' is your MultiLineString
            lines = [LineString(line) for line in merged_line.geoms]

            # Create a new GeoDataFrame with the lines
            gdf = gpd.GeoDataFrame(geometry=lines)

            # If take_longest is True, take the longest lines
            if take_longest is True:
                gdf["length"] = gdf["geometry"].length
                gdf = gdf.sort_values(by="length", ascending=False).head(longest_count)

            gdf = close_gaps(gdf, 200)

            merged_line = gdf.unary_union

            if merged_line.geom_type == "LineString":

                return merged_line
            else:
                merged_line = linemerge(merged_line)

                return merged_line


def get_longest_linestring(multilinestring):
    """
    This function returns the longest LineString from a MultiLineString.

    Arguments:
        multilinestring (MultiLineString): The MultiLineString to extract the longest LineString from

    Returns:
        LineString: The longest LineString
    """
    # If the input is a MultiLineString, return the longest LineString
    if isinstance(multilinestring, MultiLineString):
        return max(multilinestring.geoms, key=lambda line: line.length)
    # Otherwise, return the LineString
    else:
        return multilinestring


def divide_into_segments(geom, n_sect):
    """
    This function divides a geometry into n_sect segments.

    Arguments:
        geom (LineString or MultiLineString): The geometry to divide
        n_sect (int): The number of segments to divide the geometry into

    Returns:
        lines (list): The list of LineStrings that divide the geometry into n_sect segments
        mean_length (float): The mean length of the segments
        std_dev_length (float): The standard deviation of the lengths of the segments
        coeff_var_length (float): The coefficient of variation of the lengths of the segments
    """
    # Check if the geometry is None
    if geom is None:
        return None

    # Generate the points that divide the geometry into n_sect segments
    fractions = [i / n_sect for i in range(1, n_sect)]
    points = [geom.interpolate(fraction, normalized=True) for fraction in fractions]

    # Add the starting point at the beginning of the list
    points.insert(0, geom.interpolate(0, normalized=True))

    # Add the ending point at the end of the list
    points.append(geom.interpolate(1, normalized=True))

    if isinstance(geom, MultiLineString):
        # Create straight lines that connect the points
        lines = [LineString([points[i - 1], points[i]]) for i in range(1, len(points))]
    else:
        # Create lines that follow the original geometry
        lines = [
            substring(geom, start, end, normalized=True)
            for start, end in zip([0] + fractions, fractions + [1])
        ]

    # Convert the list to a GeoSeries
    line_series = gpd.GeoSeries(lines)

    # Calculate the mean length, std deviation, and coefficient of variation of the lengths
    mean_length = line_series.length.mean()
    std_dev_length = line_series.length.std()
    coeff_var_length = (std_dev_length / mean_length) * 100 if mean_length != 0 else 0

    # If the coefficient of variation is greater than 50%,
    # switch the first and last points and recreate the lines
    if coeff_var_length > 50:
        # Attempt switching first and last pints and recreating the lines
        points[0], points[-1] = points[-1], points[0]
        lines = [LineString([points[i - 1], points[i]]) for i in range(1, len(points))]
        line_series = gpd.GeoSeries(lines)

        # Recalculate the statistics
        mean_length = line_series.length.mean()
        std_dev_length = line_series.length.std()
        coeff_var_length = (
            (std_dev_length / mean_length) * 100 if mean_length != 0 else 0
        )

    return lines, mean_length, std_dev_length, coeff_var_length
