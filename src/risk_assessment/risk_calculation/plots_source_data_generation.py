"""
This script can be used to create plots from the data in the 'Bridges' GeoDataFrame.
"""

import os

import pandas as pd

from utils.find_repo import find_repo_root
from risk_assessment.risk_calculation.utils.plotting_functions import (
    create_class_column,
    calculate_monitoring_results,
    calculate_vulnerability_monitoring_results,
)


if __name__ == "__main__":

    # ====================================================
    # Define the paths and read the data
    # ====================================================

    # Find the repo root by looking for a marker file (e.g., .git)
    repo_root = find_repo_root()

    # Define the path to the data
    data_path = os.path.join(repo_root, "data")
    results_path = os.path.join(data_path, "risk_assessment")

    plots_source_data_path = os.path.join(data_path, "plots", "Source Data")

    if not os.path.exists(plots_source_data_path):
        os.makedirs(plots_source_data_path)

    # Read the bridges gdf from a pickle
    bridges_gdf = pd.read_pickle(
        os.path.join(results_path, "lsb_risk_analysis_filtered_1.35.pkl")
    )

    # Remove bridge with ID 755 that is incorrect due to lack of OSM lines
    bridges_gdf = bridges_gdf[bridges_gdf["ID"] != 755]
    # Reset the index
    bridges_gdf.reset_index(drop=True, inplace=True)

    # Print total number of bridges
    print(f"Total number of bridges: {len(bridges_gdf)}")

    # ====================================================
    # Plot PS availability
    # ====================================================

    # Create the Monitoring_PS_availability_classes column with range labels
    bridges_gdf["Monitoring_PS_availability_classes"] = pd.cut(
        bridges_gdf["Monitoring_PS_availability_only"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"],
        right=True,
        include_lowest=True,
    )

    # Save raw data
    bridges_gdf[
        ["ID", "Monitoring_PS_availability_only", "Monitoring_PS_availability_classes"]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig1a_ps_availability.csv"), index=False
    )

    # ====================================================
    # Plot SNT availability
    # ====================================================

    # Create a new column for SNT availability that has the text labels
    bridges_gdf["SNT_availability_2sc_labels"] = bridges_gdf[
        "SNT_availability_2sc"
    ].map(
        {
            1: "6 days\n 1 s/c",
            2: "6 days\n 2 s/c",
            3: "12 days\n 1 s/c",
            4: "12 days\n 2 s/c",
            0: "No SNT \n availability",
        }
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "SNT_availability_2sc",
            "SNT_availability_2sc_labels",
            "Monitoring_PS_availability_only",
        ]
    ].to_csv(os.path.join(plots_source_data_path, "fig1b_snt_avail.csv"), index=False)

    # ====================================================
    # Plot spaceborne monitoring class
    # ====================================================

    # Create the Monitoring_PS_availability_classes column with range labels
    bridges_gdf["Spaceborne monitoring class"] = pd.cut(
        bridges_gdf["Monitoring_2sc"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"],
        right=True,
        include_lowest=True,
    )

    # Save raw data
    bridges_gdf[["ID", "Monitoring_2sc", "Spaceborne monitoring class"]].to_csv(
        os.path.join(plots_source_data_path, "fig1c_space_monitoring.csv"), index=False
    )

    # ====================================================
    # Plots for PS availability by parameters
    # ====================================================

    # ========== Plot by material ==========

    # Create a new material column that combines the deck and piers/pylons materials
    # If Deck is Prestressed or Reinforced Concrete and Piers/Pylons is Reinforced Concrete,
    # then Material is Concrete
    # If both are Steel then material is Steel
    # If Deck is Steel but Piers/Pylons is Prestressed or Reinforced Concrete then Material is Composite
    # In the rest of cases the material is other

    bridges_gdf["Material"] = "Other"
    bridges_gdf.loc[
        (
            bridges_gdf["Material: Deck"].isin(
                ["Prestressed Concrete", "Reinforced Concrete"]
            )
        )
        & (bridges_gdf["Material: Piers/Pylons"] == "Reinforced Concrete"),
        "Material",
    ] = "Concrete"
    bridges_gdf.loc[
        (bridges_gdf["Material: Deck"] == "Steel")
        & (bridges_gdf["Material: Piers/Pylons"] == "Steel"),
        "Material",
    ] = "Steel"
    bridges_gdf.loc[
        (bridges_gdf["Material: Deck"] == "Steel")
        & (
            bridges_gdf["Material: Piers/Pylons"].isin(
                ["Prestressed Concrete", "Reinforced Concrete"]
            )
        ),
        "Material",
    ] = "Composite"

    print(bridges_gdf["Material"].value_counts())

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Monitoring_PS_availability_only",
            "Monitoring_PS_availability_classes",
            "Material: Deck",
            "Material: Piers/Pylons",
            # "Material",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig2a_ps_availability_by_material.csv"),
        index=False,
    )

    # ========== Plot by bridge type ==========
    bridges_gdf.rename(columns={"Type": "Bridge Type"}, inplace=True)

    bridges_gdf["Bridge Type"] = bridges_gdf["Bridge Type"].replace(
        "Cable Stayed", "Cable-Stayed"
    )
    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Monitoring_PS_availability_only",
            "Monitoring_PS_availability_classes",
            "Bridge Type",
        ]
    ].to_csv(
        os.path.join(
            plots_source_data_path, "fig2b_ps_availability_by_bridge_type.csv"
        ),
        index=False,
    )

    # ========== Plot by bridge type and segment ==========
    new_column_names = {
        "Monitoring_1_PS_availability_only": "PS_avail_1_edge",
        "Monitoring_2_PS_availability_only": "PS_avail_2_intermediate",
        "Monitoring_3_PS_availability_only": "PS_availability_central",
        "Monitoring_4_PS_availability_only": "PS_avail_4_intermediate",
        "Monitoring_5_PS_availability_only": "PS_avail_5_edge",
    }

    # Rename columns
    bridges_gdf.rename(columns=new_column_names, inplace=True)

    # Combine the edge and intermediate segments so that a mean is taken
    bridges_gdf["PS_availability_intermediate"] = (
        bridges_gdf["PS_avail_2_intermediate"] + bridges_gdf["PS_avail_4_intermediate"]
    ) / 2
    bridges_gdf["PS_availability_edge"] = (
        bridges_gdf["PS_avail_1_edge"] + bridges_gdf["PS_avail_5_edge"]
    ) / 2

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "PS_avail_1_edge",
            "PS_avail_2_intermediate",
            "PS_availability_central",
            "PS_avail_4_intermediate",
            "PS_avail_5_edge",
            "PS_availability_edge",
            "PS_availability_intermediate",
            "Bridge Type",
        ]
    ].to_csv(
        os.path.join(
            plots_source_data_path,
            "figSup1_ps_availability_by_segment_and_bridge_type.csv",
        ),
        index=False,
    )

    # ========== Plot by azimuth ==========
    # Create a new column assigning numerical categories for azimuth
    bridges_gdf["Azimuth_numerical_class"] = pd.cut(
        bridges_gdf["azimuth_3"],
        bins=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        labels=[0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0],
        right=False,
        include_lowest=True,
        ordered=False,
    )

    # Define the mapping from numerical labels to new category names
    category_mapping = {
        0: "N-S orientation",
        1: "Angled orientation",
        2: "E-W orientation",
    }

    # Map the numerical labels to the new category names
    bridges_gdf["Azimuth_numerical_class"] = bridges_gdf["Azimuth_numerical_class"].map(
        category_mapping
    )

    # Convert the new column to a categorical type with explicit categories
    bridges_gdf["Azimuth"] = pd.Categorical(
        bridges_gdf["Azimuth_numerical_class"],
        categories=[
            "N-S orientation",
            "Angled orientation",
            "E-W orientation",
        ],
        ordered=True,
    )

    bridges_gdf.rename(columns={"azimuth_3": "Azimuth_deg"}, inplace=True)

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Monitoring_PS_availability_only",
            "Monitoring_PS_availability_classes",
            "Azimuth_deg",
            "Azimuth_numerical_class",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig2c_ps_availability_by_azimuth.csv"),
        index=False,
    )

    # ====================================================
    # SHM vs spaceborne monitoring
    # ====================================================

    # ======== Plot histogram of SHM and spaceborne monitoring ======

    # Plot histogram comparing SHM and 2sc monitoring
    bridges_gdf.rename(
        columns={
            "Monitoring": "SHM-based monitoring",
            "Monitoring_2sc": "Spaceborne monitoring",
        },
        inplace=True,
    )

    # Prepare data for SHM-based monitoring
    shm_data = (
        bridges_gdf["SHM-based monitoring"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("SHM monitoring")
    )

    # Rename index values
    shm_data.index = ["No monitoring", "Very High"]

    # Prepare data for spaceborne monitoring
    spaceborne_data = (
        bridges_gdf["Spaceborne monitoring class"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("Spaceborne monitoring")
    )

    # Rename index values
    spaceborne_data.index = ["No monitoring", "Low", "Medium", "High", "Very High"]

    # Create one combined DataFrame with column for SHM and for spaceborne monitoring
    df_combined_shm_space = (
        pd.concat([shm_data, spaceborne_data], axis=1).fillna(0).transpose()
    )

    df_combined_shm_space = df_combined_shm_space[
        ["No monitoring", "Low", "Medium", "High", "Very High"]
    ]

    df_combined_shm_space = df_combined_shm_space.reindex(
        ["Spaceborne monitoring", "SHM monitoring"]
    )

    # ======== Plot overlap between SHM and spaceborne monitoring ======

    # Create column with spaceborne monitoring class
    create_class_column(
        df=bridges_gdf,
        source_col="Spaceborne monitoring",
        target_col="Spaceborne monitoring class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["No monitoring", "Low", "Medium", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    # Calculate the number of bridges in each monitoring class
    shm_counts = bridges_gdf["SHM-based monitoring"].value_counts()
    snt_counts = bridges_gdf["Spaceborne monitoring class"].value_counts()

    # Create a cross-tabulation of SHM and SNT categories
    cross_tab = pd.crosstab(
        bridges_gdf["SHM-based monitoring"], bridges_gdf["Spaceborne monitoring class"]
    )

    # Reorder columns and rows
    cross_tab = cross_tab.reindex(
        columns=["No monitoring", "Low", "Medium", "High", "Very High"], fill_value=0
    )
    cross_tab = cross_tab.reindex(index=[0, 1], fill_value=0)

    # Rename the index for clarity
    cross_tab.index = ["No SHM", "SHM"]

    # Add a SHM-based monitoring class column that has "SHM available" or "No SHM"
    bridges_gdf["SHM-based monitoring class"] = bridges_gdf["SHM-based monitoring"].map(
        {
            0: "No SHM",
            1: "SHM available",
        }
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Spaceborne monitoring",
            "Spaceborne monitoring class",
            "SHM-based monitoring",
            "SHM-based monitoring class",
        ]
    ].to_csv(
        os.path.join(
            plots_source_data_path, "fig3b_SHM_and_space_monitoring_overlap.csv"
        ),
        index=False,
    )

    # ======== Plot monitoring per continent  ======

    # Generate a table that shows the number of bridges in each monitoring class for each continent
    # Last row should have the values for all bridges (globally)

    # Calculate the number of bridges in each monitoring class for each continent
    monitoring_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["SHM-based monitoring", "Spaceborne monitoring class"],
        index_col="Region",
    )

    # Rename columns
    monitoring_results.rename(columns={0: "No SHM", 1: "SHM"}, inplace=True)

    # Reorder columns
    desired_order = [
        "No SHM",
        "SHM",
        "No monitoring",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]
    monitoring_results = monitoring_results.reindex(columns=desired_order)

    # Create a new DataFrame for the additional table
    combined_monitoring_results = pd.DataFrame(index=monitoring_results.index)
    combined_monitoring_results["No_SHM"] = monitoring_results["No SHM"]
    combined_monitoring_results["SHM"] = monitoring_results["SHM"]
    combined_monitoring_results["SNT_unlikely"] = monitoring_results[
        ["No monitoring", "Low"]
    ].sum(axis=1)
    combined_monitoring_results["SNT_possible"] = monitoring_results[
        ["Medium", "High", "Very High"]
    ].sum(axis=1)

    # Create a horizontal histogram that shows the share of bridges in each monitoring class for each continent
    # There should be two labels, one for continent and one for the total count of bridges
    # Then, there should be two plots next to each other, one for SHM and one for SNT

    shm_monitoring_by_continent = combined_monitoring_results[["No_SHM", "SHM"]]
    space_monitoring_by_continent = combined_monitoring_results[
        ["SNT_unlikely", "SNT_possible"]
    ]

    # Add total count of bridges
    shm_monitoring_by_continent = shm_monitoring_by_continent.copy()
    shm_monitoring_by_continent.loc[:, "Total count of bridges"] = (
        shm_monitoring_by_continent.sum(axis=1)
    )

    # Normalize the data to get the share of bridges in each monitoring class
    shm_monitoring_by_continent[["No_SHM", "SHM"]] = shm_monitoring_by_continent[
        ["No_SHM", "SHM"]
    ].div(shm_monitoring_by_continent[["No_SHM", "SHM"]].sum(axis=1), axis=0)

    space_monitoring_by_continent = space_monitoring_by_continent.div(
        space_monitoring_by_continent.sum(axis=1), axis=0
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "SHM-based monitoring",
            "SHM-based monitoring class",
            "Spaceborne monitoring",
            "Spaceborne monitoring class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig3a_monitoring_by_continent.csv"),
        index=False,
    )

    # ===== Structural vulnerability distribution by region =====

    # Create a new column for Vulnerability_norm_class
    bridges_gdf["Vulnerability_norm_class"] = pd.cut(
        bridges_gdf["Vulnerability_norm"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=True,
        include_lowest=True,
    )

    # Generate a table that shows the number of bridges in each vulnerability class for each continent
    # Last row should have the values for all bridges (globally)

    # Calculate the number of bridges in each vulnerability class for each continent
    vulnerability_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["Vulnerability_norm_class"],
        index_col="Region",
    )

    # Reorder columns
    desired_order = [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]
    vulnerability_results = vulnerability_results.reindex(columns=desired_order)

    # Add total count of bridges
    vulnerability_results = vulnerability_results.copy()
    vulnerability_results.loc[:, "Total count of bridges"] = vulnerability_results.sum(
        axis=1
    )

    # Normalize the data to get the share of bridges in each monitoring class
    vulnerability_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ] = vulnerability_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ].div(
        vulnerability_results[
            [
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High",
            ]
        ].sum(axis=1),
        axis=0,
    )

    vulnerability_results.rename(
        index={
            "North America": "North America",
            "Latin America/Caribbean": "Latin America \n/Caribbean",
        },
        inplace=True,
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Vulnerability_norm",
            "Vulnerability_norm_class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig4a_vulnerability_by_continent.csv"),
        index=False,
    )

    # ======== Monitoring and structural vulnerability  ======
    # Define the order of vulnerabilities
    vulnerability_order = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Calculate the number of bridges in each monitoring class for each vulnerability level
    monitoring_results_vulnerability = calculate_vulnerability_monitoring_results(
        df=bridges_gdf,
        vulnerability_order=vulnerability_order,
        vulnerability_col="Vulnerability_norm",
        monitoring_cols=["SHM-based monitoring", "Spaceborne monitoring class"],
    )

    monitoring_results_vulnerability.rename(
        columns={
            "0": "SHM 0",
            "1": "SHM 1",
            "Very High": "SNT Very High",
            "No monitoring": "SNT No monitoring",
            "Low": "SNT Low",
            "Medium": "SNT Medium",
            "High": "SNT High",
        },
        inplace=True,
    )

    # Reorder columns
    desired_order = [
        "SHM 0",
        "SHM 1",
        "SNT No monitoring",
        "SNT Low",
        "SNT Medium",
        "SNT High",
        "SNT Very High",
    ]
    monitoring_results_vulnerability = monitoring_results_vulnerability.reindex(
        columns=desired_order
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Vulnerability_norm",
            "Vulnerability_norm_class",
            "SHM-based monitoring",
            "SHM-based monitoring class",
            "Spaceborne monitoring",
            "Spaceborne monitoring class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig4b_monitoring_vulnerability.csv"),
        index=False,
    )

    # ====================================================
    # Risk analysis
    # ====================================================

    # ===== Risk distribution by region =====

    # Create a new column for Multi-hazard_risk_shm_class
    bridges_gdf["Multi-hazard_risk_shm_class"] = pd.cut(
        bridges_gdf["Multi-hazard_risk_shm"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=True,
        include_lowest=True,
    )

    # Generate a table that shows the number of bridges in each risk class for each continent
    # Last row should have the values for all bridges (globally)

    # Calculate the number of bridges in each risk class for each continent
    risk_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["Multi-hazard_risk_shm_class"],
        index_col="Region",
    )

    # Reorder columns
    desired_order = [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]
    risk_results = risk_results.reindex(columns=desired_order)

    # Add total count of bridges
    risk_results = risk_results.copy()
    risk_results.loc[:, "Total count of bridges"] = risk_results.sum(axis=1)

    # Normalize the data to get the share of bridges in each monitoring class
    risk_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ] = risk_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ].div(
        risk_results[
            [
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High",
            ]
        ].sum(axis=1),
        axis=0,
    )

    risk_results.rename(
        index={
            "North America": "North America",
            "Latin America/Caribbean": "Latin America \n/Caribbean",
        },
        inplace=True,
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Multi-hazard_risk_shm",
            "Multi-hazard_risk_shm_class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig5a_risk_by_continent.csv"),
        index=False,
    )

    # ===== Monitoring and multi-hazard risk =====
    create_class_column(
        df=bridges_gdf,
        source_col="SHM-based monitoring",
        target_col="Monitoring_shm_class",
        bins=[0, 0.5, 1.01],
        labels=["No SHM monitoring", "SHM monitoring"],
        right=False,
        include_lowest=True,
    )

    pivot_table_shm = bridges_gdf.pivot_table(
        index="Multi-hazard_risk_shm_class",
        columns="Monitoring_shm_class",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )

    # Prepare the data for combined monitoring
    bridges_gdf["Monitoring_combined_class"] = bridges_gdf.apply(
        lambda row: (
            row["Monitoring_shm_class"]
            if row["Monitoring_shm_class"] == "SHM monitoring"
            else row["Spaceborne monitoring class"]
        ),
        axis=1,
    )

    # Create a new column for Multi-hazard_risk_com_class
    bridges_gdf["Multi-hazard_risk_com_class"] = pd.cut(
        bridges_gdf["Multi-hazard_risk_com"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=True,
        include_lowest=True,
    )

    pivot_table_combined_monitoring = bridges_gdf.pivot_table(
        index="Multi-hazard_risk_com_class",
        columns="Monitoring_combined_class",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )

    # Combine the data into a single DataFrame
    combined_risk = pd.concat(
        [pivot_table_shm, pivot_table_combined_monitoring], axis=1
    ).fillna(0)

    # Remove duplicated column (SHM monitoring)
    combined_risk = combined_risk.loc[:, ~combined_risk.columns.duplicated()]

    # Rename columns by adding "SNT" at the beginning
    combined_risk.rename(
        columns={
            "Very High": "SNT Very High",
            "Low": "SNT Low",
            "Medium": "SNT Medium",
            "High": "SNT High",
        },
        inplace=True,
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Multi-hazard_risk_shm",
            "Multi-hazard_risk_shm_class",
            "Multi-hazard_risk_com",
            "Multi-hazard_risk_com_class",
            "SHM-based monitoring",
            "SHM-based monitoring class",
            "Spaceborne monitoring",
            "Spaceborne monitoring class",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig5b_monitoring_risk.csv"),
        index=False,
    )

    # ==== Change in risk =====

    # Calculate the number of bridges in each monitoring class for each continent
    multi_hazard_shm_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["Multi-hazard_risk_shm_class"],
        index_col="Region",
    )

    multi_hazard_com_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["Multi-hazard_risk_com_class"],
        index_col="Region",
    )

    # Reorder columns
    desired_order = [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]
    multi_hazard_shm_results = multi_hazard_shm_results.reindex(columns=desired_order)
    multi_hazard_com_results = multi_hazard_com_results.reindex(columns=desired_order)

    # Remove row with index equal to "Global"
    multi_hazard_shm_results = multi_hazard_shm_results.drop("Global")
    multi_hazard_com_results = multi_hazard_com_results.drop("Global")

    # Define the risk categories
    risk_categories = ["Very Low", "Low", "Medium", "High", "Very High"]

    # ====================================================
    # Risk change map
    # ====================================================

    # Get the mean risk for each region for SHM and combined monitoring
    mean_risk_shm = bridges_gdf.groupby("Region")["Multi-hazard_risk_shm"].mean()
    mean_risk_shm.loc["Global"] = bridges_gdf["Multi-hazard_risk_shm"].mean()

    mean_risk_com = bridges_gdf.groupby("Region")["Multi-hazard_risk_com"].mean()
    mean_risk_com.loc["Global"] = bridges_gdf["Multi-hazard_risk_com"].mean()

    # Calculate the difference
    risk_difference = mean_risk_com - mean_risk_shm

    # Calculate the percentage change
    risk_percentage_change = (risk_difference / mean_risk_shm) * 100

    # Calculate the individual changes in risk for each bridge
    bridges_gdf["individual_percentage_change"] = (
        (bridges_gdf["Multi-hazard_risk_com"] - bridges_gdf["Multi-hazard_risk_shm"])
        / bridges_gdf["Multi-hazard_risk_shm"]
        * 100
    )

    # Calculate the average of the individual changes for each region
    average_individual_change = bridges_gdf.groupby("Region")[
        "individual_percentage_change"
    ].mean()
    average_individual_change.loc["Global"] = bridges_gdf[
        "individual_percentage_change"
    ].mean()

    # Add a "Global" region to each row in the DataFrame
    bridges_gdf_global = bridges_gdf.copy()
    bridges_gdf_global["Region"] = "Global"

    # Concatenate the original DataFrame with the global DataFrame
    bridges_gdf_with_global = pd.concat(
        [bridges_gdf, bridges_gdf_global], ignore_index=True
    )

    # Plot the whisker box plot
    # Reshape the data for Seaborn
    long_df = pd.melt(
        bridges_gdf_with_global,
        id_vars=["Region"],
        value_vars=["Multi-hazard_risk_shm", "Multi-hazard_risk_com"],
        var_name="Risk Type",
        value_name="Risk Value",
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Multi-hazard_risk_shm",
            "Multi-hazard_risk_com",
            "Multi-hazard_risk_shm_class",
            "Multi-hazard_risk_com_class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig6a_box_plot_risk.csv"),
        index=False,
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Multi-hazard_risk_shm",
            "Multi-hazard_risk_com",
            "Region",
            "individual_percentage_change",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig6b_map_relative_change.csv"),
        index=False,
    )

    # Plot map with bridge locations

    # Save raw data
    bridges_gdf[["ID", "Region", "geometry"]].to_csv(
        os.path.join(plots_source_data_path, "figSup2_map_bridges_locations.csv"),
        index=False,
    )

    # ====================================================
    # Change in Sentinel-1 availability
    # ====================================================

    # Calculate the change in Sentinel-1 availability
    snt_1sc_availability = bridges_gdf["SNT_availability_1sc"].value_counts()
    snt_2sc_availability = bridges_gdf["SNT_availability_2sc"].value_counts()
    df_snt_availability = pd.DataFrame(
        {
            "Sentinel-1 availability \n when only 1 s/c on orbit": snt_1sc_availability,
            "Sentinel-1 availability \n when 2 s/c on orbit": snt_2sc_availability,
        }
    ).fillna(0)

    # Rename the index to make it more readable
    df_snt_availability.rename(
        index={
            0: "No SNT \n availability",
            1: "6 days\n 1 s/c",
            2: "6 days\n 2 s/c",
            3: "12 days\n 1 s/c",
            4: "12 days\n 2 s/c",
        },
        inplace=True,
    )

    # Define the desired order of categories
    desired_order = [
        "6 days\n 2 s/c",
        "12 days\n 2 s/c",
        "6 days\n 1 s/c",
        "12 days\n 1 s/c",
        "No SNT \n availability",
    ]

    # Reindex the DataFrame to match the desired order
    df_snt_availability = df_snt_availability.reindex(desired_order, fill_value=0)

    # Calculate the percentage for each category
    df_snt_availability_percentages = df_snt_availability.div(
        df_snt_availability.sum(axis=0), axis=1
    )

    df_snt_availability_percentages = df_snt_availability_percentages.transpose()

    df_snt_availability_percentages.rename(
        index={
            "Sentinel-1 availability \n when only 1 s/c on orbit": "Sentinel-1 availability\nwhen 1 satellite on orbit",
            "Sentinel-1 availability \n when 2 s/c on orbit": "Sentinel-1 availability\nwhen 2 satellites on orbit",
        },
        inplace=True,
    )

    bridges_gdf["SNT_availability_1sc_labels"] = bridges_gdf[
        "SNT_availability_1sc"
    ].map(
        {
            1: "6 days\n 1 s/c",
            2: "6 days\n 2 s/c",
            3: "12 days\n 1 s/c",
            4: "12 days\n 2 s/c",
            0: "No SNT \n availability",
        }
    )

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "SNT_availability_1sc",
            "SNT_availability_2sc",
            "SNT_availability_1sc_labels",
            "SNT_availability_2sc_labels",
            "Monitoring_PS_availability_only",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig7a_sentinel_availability.csv"),
        index=False,
    )

    # ===== Plot the change in Sentinel-1 availability on a map =====
    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 4)
        & (bridges_gdf["SNT_availability_1sc"] == 3),
        "SNT_availability_change",
    ] = "Lost 1 s/c"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 1)
        & (bridges_gdf["SNT_availability_1sc"] == 3),
        "SNT_availability_change",
    ] = "Increased time"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 2)
        & (bridges_gdf["SNT_availability_1sc"] == 4),
        "SNT_availability_change",
    ] = "Increased time"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 3)
        & (bridges_gdf["SNT_availability_1sc"] == 3),
        "SNT_availability_change",
    ] = "No change"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 4)
        & (bridges_gdf["SNT_availability_1sc"] == 4),
        "SNT_availability_change",
    ] = "No change"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 3)
        & (bridges_gdf["SNT_availability_1sc"] == 0),
        "SNT_availability_change",
    ] = "Availability lost completely"

    bridges_gdf.loc[
        (bridges_gdf["SNT_availability_2sc"] == 3)
        & (bridges_gdf["SNT_availability_1sc"] == 4),
        "SNT_availability_change",
    ] = "Increased s/c availability"

    # Count how availability changed
    snt_change_counted = bridges_gdf["SNT_availability_change"].value_counts()

    # Define the mapping dictionary
    snt_availability_change_mapping = {
        "Increased time": 1,
        "No change": 2,
        "Lost 1 s/c": 3,
        "Increased s/c availability": 4,
        "Availability lost completely": 5,
    }

    reverse_mapping = {v: k for k, v in snt_availability_change_mapping.items()}

    # Create a new column with numerical values
    bridges_gdf["SNT_availability_change_num"] = bridges_gdf[
        "SNT_availability_change"
    ].map(snt_availability_change_mapping)

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "SNT_availability_1sc",
            "SNT_availability_2sc",
            "SNT_availability_change_num",
            "SNT_availability_1sc_labels",
            "SNT_availability_2sc_labels",
            "SNT_availability_change",
            "Region",
            "geometry",
        ]
    ].to_csv(
        os.path.join(plots_source_data_path, "fig7b_map_snt_availability_change.csv"),
        index=False,
    )

    # ====================================================
    # Create monitoring factor sensitivity analysis data
    # ====================================================
    # Define the factors to analyze
    factors = [1.15, 1.25, 1.35, 1.45, 1.55]

    # Initialize a dictionary to store data from all factors
    sensitivity_data = {}

    # Read data for each factor
    for factor in factors:
        # Read the bridges gdf from pickle for this factor
        factor_bridges_gdf = pd.read_pickle(
            os.path.join(results_path, f"lsb_risk_analysis_filtered_{factor}.pkl")
        )

        # Remove bridge with ID 755 (same as main analysis)
        factor_bridges_gdf = factor_bridges_gdf[factor_bridges_gdf["ID"] != 755]

        # Store the risk values for this factor
        sensitivity_data[str(factor)] = factor_bridges_gdf.set_index("ID")[
            "Multi-hazard_risk_com"
        ]

        # Store ID and Region from the first iteration (they should be the same across all files)
        if factor == factors[0]:
            base_data = factor_bridges_gdf[["ID", "Region"]].copy()

    # Create the sensitivity analysis dataframe
    sensitivity_df = base_data.copy()

    # Add risk columns for each factor
    for factor in factors:
        sensitivity_df[str(factor)] = sensitivity_df["ID"].map(
            sensitivity_data[str(factor)]
        )

    # Sort by ID for consistency
    sensitivity_df = sensitivity_df.sort_values("ID").reset_index(drop=True)

    # Save the sensitivity analysis data
    sensitivity_df.to_csv(
        os.path.join(
            plots_source_data_path, "figSup5_monitoring_factor_sensitivity.csv"
        ),
        index=False,
    )
