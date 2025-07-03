"""
This script is used to identify bridges that are very close to each other
and are potentially the same bridge. It reads the LSB Database CSV file
and calculates the distance between each combination of points. If the
distance between two points is less than 0.01 degrees, it prints the IDs
of the two bridges and the distance between them.
"""

import os
import pandas as pd
from itertools import combinations

data_path = os.path.join("/mnt/g/RISK_PAPER/Vulnerability")
lsb_path = os.path.join(data_path, "LSB Database.csv")

# Read the CSV file into a DataFrame
df = pd.read_csv(lsb_path)

# Calculate the distance between each combination of points
distances = []
for point1, point2 in combinations(df.index, 2):
    lat1, lon1 = df.loc[point1, ["Latitude", "Longitude"]]
    lat2, lon2 = df.loc[point2, ["Latitude", "Longitude"]]
    distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
    distances.append(distance)

    if distance < 0.01:
        print(
            "Bridge with ID:",
            df.iloc[point1]["ID"],
            "seems to be very close to bridge with ID:",
            df.iloc[point2]["ID"],
            "with a distance between them being:",
            distance,
        )

# Add the distances to the DataFrame
df["Distance"] = distances

# Filter the DataFrame to include only rows with distance less than 0.01 deg
filtered_df = df[df["Distance"] < 0.01]

# Print the filtered rows
print(filtered_df)
