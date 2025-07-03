"""
Author: Dominika Malinowska

This script defines functions for assessing exposure of bridges based on the data from the LSB Database.
The methodology uses the type of road that is found on the bridge to assign an exposure class.
"""


def get_exposure(df):
    """
    This function assigns an exposure class to each bridge in the dataframe based on the type of road on the bridge.

    Parameters:
        df (DataFrame): The dataframe containing the bridge data.
                        It should have columns for 'Road', 'Rail', and 'highway'.

    Returns:
        DataFrame:  The input dataframe with an additional 'Exposure' column
                    containing the exposure class for each bridge.
    """

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Print the index for debugging purposes
        # print(index)

        # Check the type of road on the bridge and assign an exposure class accordingly
        if row["Road"] == 1 and row["Rail"] == 1:
            # If the bridge has both a road and a rail, assign the highest exposure class
            df.at[index, "Exposure"] = 5
        elif row["Rail"] == 1:
            # If the bridge only has a rail, assign the second highest exposure class
            df.at[index, "Exposure"] = 4
        elif isinstance(row["highway"], str) and "motorway" in row["highway"]:
            # If the bridge is a motorway, assign the second highest exposure class
            df.at[index, "Exposure"] = 4
        elif isinstance(row["highway"], str) and "trunk" in row["highway"]:
            # If the bridge is a trunk road, assign the middle exposure class
            df.at[index, "Exposure"] = 3
        elif isinstance(row["highway"], str) and "primary" in row["highway"]:
            # If the bridge is a primary road, assign the middle exposure class
            df.at[index, "Exposure"] = 3
        elif isinstance(row["highway"], str) and "secondary" in row["highway"]:
            # If the bridge is a secondary road, assign the second lowest exposure class
            df.at[index, "Exposure"] = 2
        else:
            # If the bridge is none of the above, assign the lowest exposure class
            df.at[index, "Exposure"] = 1

    # Return the dataframe with the added 'Exposure' column
    return df
