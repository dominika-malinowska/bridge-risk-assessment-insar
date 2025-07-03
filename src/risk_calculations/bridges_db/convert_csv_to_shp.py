# Convert csv to shp

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Read csv
df = pd.read_csv(
    "/mnt/g/RISK_PAPER/lsb_bridges_database/LSB Database_corrected.csv",
    # encoding="utf-8-sig",
)

# Remove rows with missing coordinates
df = df.dropna(subset=["Latitude", "Longitude"])

# Print column names to verify
# print("Column names:", df.columns)

# Ensure column names are stripped of leading/trailing spaces
df.columns = df.columns.str.strip()

# Remove spaces in column names
df.columns = df.columns.str.replace(" ", "").str.replace(r"[^a-zA-Z0-9_]", "")

# # Check for missing values and fill or drop them
# df = df.fillna("")  # Fill missing values with an empty string or appropriate value

# Create geometry
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

# # Assign data types to columns
gdf["ID"] = gdf["ID"].astype(int)
gdf["ConstructionStart"] = gdf["ConstructionStart"].astype(int)
gdf["ConstructionFinished"] = gdf["ConstructionFinished"].astype(int)
gdf["Latitude"] = gdf["Latitude"].astype(float)
gdf["Longitude"] = gdf["Longitude"].astype(float)

# # Ensure all string columns are properly encoded
# for col in gdf.select_dtypes(include=["object"]).columns:
#     gdf[col] = gdf[col].apply(
#         lambda x: x.encode("utf-8", errors="ignore").decode("utf-8")
#     )


# Check data types
print("Data types:", gdf.dtypes)

# Save to shp
gdf.to_file(
    "/mnt/g/RISK_PAPER/lsb_bridges_database/LSB Database_corrected.shp",
    # encoding="utf-8",
)
