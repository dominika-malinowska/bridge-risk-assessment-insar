import os
import json


def generate_json(region, gdf, path_output, json_path):
    """
    Generate json files used to cut the OSM data into tiles.
    """

    # Initialize the dictionary with the fixed fields
    data = {
        "directory": "{}".format(path_output),
        "extracts": [],
    }

    # Initialize the file counter
    file_counter = 1

    # Iterate over the rows of the GeoDataFrame
    for i, row in gdf.iterrows():
        # Get the bounding box of the geometry
        bbox = list(row["geometry"].bounds)

        # Extract the coordinates of the polygon
        coords = row["geometry"].exterior.coords.xy

        # Initialize min and max coordinates
        min_x, min_y = min(coords[0]), min(coords[1])
        # max_x, max_y = max(coords[0]), max(coords[1])

        # Convert min coordinates to integers
        lon = int(min_x)
        lat = int(min_y)

        # Format latitude and longitude
        lat_f = (
            "N{:02d}".format(lat + 1) if lat >= 0 else "S{:02d}".format(abs(lat + 1))
        )
        lon_f = "E{:03d}".format(lon) if lon >= 0 else "W{:03d}".format(abs(lon))

        # Create a new dictionary for this extract
        extract = {
            "output": "{}{}.osm.pbf".format(lat_f, lon_f),
            "bbox": bbox,
        }

        # Add the new extract to the list
        data["extracts"].append(extract)

        # If we've added 50 extracts, write the current data to a file and start a new one
        if (i + 1) % 50 == 0:
            # Convert the dictionary to a JSON string
            json_str = json.dumps(data, indent=4)

            # Define the path to the output file
            output_file_path = os.path.join(
                json_path, f"polygons_{region}_{file_counter}.json"
            )

            # Write the JSON string to a file
            with open(output_file_path, "w") as f:
                f.write(json_str)

            # Reset the data dictionary and increment the file counter
            data["extracts"] = []
            file_counter += 1

    # Write any remaining extracts to a final file
    if data["extracts"]:
        # Convert the dictionary to a JSON string
        json_str = json.dumps(data, indent=4)

        # Define the path to the output file
        output_file_path = os.path.join(
            json_path, f"polygons_{region}_{file_counter}.json"
        )

        # Write the JSON string to a file
        with open(output_file_path, "w") as f:
            f.write(json_str)


def cut_osm(osmium_path, json_path, regional_osm):

    # Read all json files in the json directory
    json_files = [f for f in os.listdir(json_path) if f.endswith(".json")]

    # Iterate over the json files

    for polygon_file in json_files:
        # Construct the path to the polygon file
        polygon_file = os.path.join(json_path, polygon_file)

        # print(f"{osmium_path} extract -c {polygon_file} {regional_osm} --overwrite")

        # Cut the OSM data using the polygon file
        os.system(f"{osmium_path} extract -c {polygon_file} {regional_osm} --overwrite")
