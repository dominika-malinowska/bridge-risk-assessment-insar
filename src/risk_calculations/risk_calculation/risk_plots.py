"""
This script can be used to create plots from the data in the 'Bridges' GeoDataFrame.
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


from ps_predictions.second_paper.risk_calculation.utils.plotting_functions import (
    plot_map,
    plot_histogram,
    plot_multiple_histograms,
    plot_pie_chart,
    plot_bridge_parameter_histogram,
    calculate_counts,
    calculate_percentages,
    save_to_csv,
    save_to_excel,
    create_class_column,
    calculate_monitoring_results,
    save_table_as_image,
    calculate_vulnerability_monitoring_results,
    plot_stacked_histogram,
    plot_stacked_histogram_new,
)


if __name__ == "__main__":

    # ====================================================
    # Define the paths and read the data
    # ====================================================

    # Define the path to the data
    data_path = "/mnt/g/RISK_PAPER"

    plots_path = os.path.join(data_path, "plots_new_data")

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Read the bridges gdf from a pickle
    bridges_gdf = pd.read_pickle(
        os.path.join(data_path, "lsb_risk_analysis_filtered.pkl")
        # os.path.join(data_path, "lsb_risk_analysis.pkl")
    )

    # Set the font size
    plt.rcParams["font.size"] = 18

    # ====================================================
    # Plot on maps
    # ====================================================

    col_dict = {
        "Exposure_norm": "Exposure",
        "Vulnerability_norm": "Vulnerability (monitoring not included)",
        "ls_max": "Landslide hazard",
        "su_max": "Subsidence hazard",
        "Multi-hazard": "Multi-hazard (subsidence and landslide)",
        "Landslide_risk_without_monitoring": "Landslide risk (monitoring not included)",
        "Subsidence_risk_without_monitoring": "Subsidence risk (monitoring not included)",
        "Multi-hazard_risk_without_monitoring": "Multi-hazard risk (monitoring not included)",
        "Monitoring": "SHM Monitoring",
        "Monitoring_PS_availability_only": (
            "Monitoring based on PS availability only \n"
            "(assuming perfect SNT availability, every 6 days with 2 s/c)"
        ),
        "SNT_availability_1sc": "SNT-1 availability after failure (only one s/c available)",
        "SNT_availability_2sc": "SNT-1 availability before failure (i.e. when both s/c were available)",
        "Monitoring_1sc": "Monitoring class (1 s/c)",
        "Monitoring_2sc": "Monitoring class (2 s/c)",
        "Vulnerability_norm_monitoring_shm": "Vulnerability (monitoring with SHM included)",
        "Vulnerability_norm_monitoring_1sc": "Vulnerability (monitoring with 1 s/c included)",
        "Vulnerability_norm_monitoring_2sc": "Vulnerability (monitoring with 2 s/c included)",
        "Landslide_risk_shm": "Landslide risk (monitoring with shm included)",
        "Landslide_risk_1sc": "Landslide risk (monitoring with 1 s/c included)",
        "Landslide_risk_2sc": "Landslide risk (monitoring with 2 s/c included)",
        "Subsidence_risk_shm": "Subsidence risk (monitoring with shm included)",
        "Subsidence_risk_1sc": "Subsidence risk (monitoring with 1 s/c included)",
        "Subsidence_risk_2sc": "Subsidence risk (monitoring with 2 s/c included)",
        "Multi-hazard_risk_shm": "Multi-hazard risk (monitoring with shm included)",
        "Multi-hazard_risk_1sc": "Multi-hazard risk (monitoring with 1 s/c included)",
        "Multi-hazard_risk_2sc": "Multi-hazard risk (monitoring with 2 s/c included)",
        "Multi-hazard_risk_com": "Multi-hazard risk (combined SHM and 2 s/c)",
    }

    color_dict = {
        0.1: "#4575b4",
        0.3: "#91bfdb",
        0.5: "#fee090",
        0.7: "#fc8d59",
        0.9: "#d73027",
    }
    color_dict_monitoring = {
        0.1: "#d73027",
        0.3: "#fc8d59",
        0.5: "#fee090",
        0.7: "#91bfdb",
        0.9: "#4575b4",
    }

    color_dict_snt_avail = {
        0: "#648fff",
        1: "#785ef0",
        2: "#dc267f",
        3: "#fe6100",
        4: "#ffb000",
    }

    bounds = [-0.0001, 0.201, 0.401, 0.601, 0.801, 1.0001]
    bounds_snt_avail = [0, 1, 2, 3, 4, 5]

    tick_labels_risk = ["Very Low", "Low", "Medium", "High", "Very High"]
    tick_labels_monitoring = [
        "Very Low/No monitoring",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]
    tick_labels_snt_avail = [
        "No availability",
        "6 days, 1 s/c",
        "6 days, 2 s/c",
        "12 days, 1 s/c",
        "12 days, 2 s/c",
    ]
    tick_locations = [0.1, 0.3, 0.5, 0.7, 0.9]
    tick_locations_snt_avail = [0.5, 1.5, 2.5, 3.5, 4.5]

    # Assuming bridges_gdf and plots_path are defined
    for col_title, plot_title in col_dict.items():
        if col_title.startswith("Monitoring"):
            plot_map(
                bridges_gdf,
                col_title,
                plot_title,
                color_dict_monitoring,
                bounds,
                tick_labels_monitoring,
                tick_locations,
                plots_path,
            )
        elif col_title.startswith("SNT"):
            bridges_gdf[col_title] = bridges_gdf[col_title].fillna(0)
            plot_map(
                bridges_gdf,
                col_title,
                plot_title,
                color_dict_snt_avail,
                bounds_snt_avail,
                tick_labels_snt_avail,
                tick_locations_snt_avail,
                plots_path,
            )
        else:
            plot_map(
                bridges_gdf,
                col_title,
                plot_title,
                color_dict,
                bounds,
                tick_labels_risk,
                tick_locations,
                plots_path,
            )

    # ====================================================
    # Generate tables with counts and percentages
    # of the same variable that were plotted on maps and save to csv
    # ====================================================

    # For each col_title in the col_dict, create a table that in rows has continents
    # and in columns has the levels of the column, and the values are the number of bridges
    # that fall into that category. The categories have to be binned into the following bins:
    # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0

    # Define the bins and labels
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    # Assuming 'continent' is a column in bridges_gdf
    continents = bridges_gdf["Region"].unique()

    # Initialize an empty dictionary to store the results
    monitoring_results_counts = {}
    monitoring_results_percentages = {}

    # Calculate counts and percentages for each column title
    for col_title in col_dict.keys():
        df_counts = calculate_counts(
            bridges_gdf, col_title, bins, bin_labels, continents
        )
        monitoring_results_counts[col_title] = df_counts

        df_percentages = calculate_percentages(
            bridges_gdf, col_title, bins, bin_labels, continents
        )
        monitoring_results_percentages[col_title] = df_percentages

    # Save counts to CSV
    df_list_counts = [
        df.assign(col_title=col_title)
        for col_title, df in monitoring_results_counts.items()
    ]
    save_to_csv(df_list_counts, os.path.join(plots_path, "combined_counts.csv"))

    # Save percentages to Excel
    save_to_excel(
        monitoring_results_percentages,
        os.path.join(plots_path, "combined_percentages.xlsx"),
    )
    # ====================================================
    # Plot histograms counting bridges in a category
    # ====================================================

    # Plot histogram comparing SHM and 2sc monitoring
    columns_to_plot = [
        "Monitoring",
        # "Monitoring_1sc",
        "Monitoring_2sc",
    ]
    for col in columns_to_plot:
        plot_histogram(
            bridges_gdf,
            np.arange(0, 1.1, 0.2).tolist(),
            col,
            0.9,
            plots_path,
        )

    plot_multiple_histograms(
        bridges_gdf,
        np.arange(0, 1.1, 0.2).tolist(),
        columns_to_plot,
        0.9,
        plots_path,
    )

    # Plot histogram comparing SNT_availability before and after SNT-1B failure
    columns_to_plot = [
        "SNT_availability_1sc",
        "SNT_availability_2sc",
    ]
    for col in columns_to_plot:
        plot_histogram(
            bridges_gdf,
            np.arange(0, 6, 1).tolist(),
            col,
            0.8,
            plots_path,
        )

    plot_multiple_histograms(
        bridges_gdf,
        np.arange(0, 6, 1).tolist(),
        columns_to_plot,
        0.8,
        plots_path,
    )

    # Plot histograms comparing risk values when monitored with SHM and 2sc
    columns_to_plot = [
        "Multi-hazard_risk_shm",
        # "Multi-hazard_risk_1sc",
        "Multi-hazard_risk_2sc",
    ]
    for col in columns_to_plot:
        plot_histogram(
            bridges_gdf,
            np.arange(0, 1.1, 0.1).tolist(),
            col,
            0.4,
            plots_path,
        )

    plot_multiple_histograms(
        bridges_gdf,
        np.arange(0, 1.1, 0.1).tolist(),
        columns_to_plot,
        0.4,
        plots_path,
    )

    # Plot histograms comparing the difference in risk values between SHM and 2sc monitoring
    bridges_gdf["diff_in_risk_shm_vs_2sc"] = (
        bridges_gdf["Multi-hazard_risk_shm"] - bridges_gdf["Multi-hazard_risk_2sc"]
    )

    bridges_gdf["diff_in_risk_1sc_vs_2sc"] = (
        bridges_gdf["Multi-hazard_risk_1sc"] - bridges_gdf["Multi-hazard_risk_2sc"]
    )

    columns_to_plot = ["diff_in_risk_shm_vs_2sc", "diff_in_risk_1sc_vs_2sc"]
    plot_multiple_histograms(
        bridges_gdf,
        np.arange(0, 0.05, 0.005).tolist(),
        columns_to_plot,
        0.8,
        plots_path,
    )

    # ====================================================
    # Create pie charts showing the share of bridges in each monitoring class
    # ====================================================

    # Create two pie charts one that shows the share of bridges monitored with 1sc
    # and and other for 2sc. Have a separate category
    # depending on the time between acquisition and whether one or two s/c are available

    # Define the mapping of original values to desired labels
    label_mapping = {
        0: "No SNT availability",
        1: "6 days, 1 s/c",
        2: "6 days, 2 s/c",
        3: "12 days, 1 s/c",
        4: "12 days, 2 s/c",
    }

    # Define the mapping of labels to colors
    color_mapping = {
        "No SNT availability": "#ff9999",
        "6 days, 1 s/c": "#66b3ff",
        "6 days, 2 s/c": "#99ff99",
        "12 days, 1 s/c": "#ffcc99",
        "12 days, 2 s/c": "#c2c2f0",
    }

    # Get the value counts and plot the pie chart for SNT_availability_1sc
    snt_1sc_availability = bridges_gdf["SNT_availability_1sc"].value_counts()
    plot_pie_chart(
        snt_1sc_availability,
        "SNT availability over bridges \n after S1B failure (only one s/c available)",
        "pie_chart_snt_availability_1sc.jpg",
        plots_path,
        label_mapping,
        color_mapping,
        sort_order=[0, 3, 4],  # Specify the desired order here
    )

    # Get the value counts and plot the pie chart for SNT_availability_2sc
    snt_2sc_availability = bridges_gdf["SNT_availability_2sc"].value_counts()
    plot_pie_chart(
        snt_2sc_availability,
        "SNT availability over bridges \n before S1B failure (two s/c available)",
        "pie_chart_snt_availability_2sc.jpg",
        plots_path,
        label_mapping,
        color_mapping,
        sort_order=[1, 2, 3, 4],  # Specify the desired order here
    )

    # Make histogram instead of pie chart
    # Combine the value counts into a single DataFrame
    combined_counts = pd.DataFrame(
        {
            "SNT_availability_1sc": snt_1sc_availability,
            "SNT_availability_2sc": snt_2sc_availability,
        }
    ).fillna(0)

    # Rename the index to make it more readable
    combined_counts.rename(
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
    combined_counts = combined_counts.reindex(desired_order, fill_value=0)

    # Calculate the percentage for each category
    combined_percentages = combined_counts.div(combined_counts.sum(axis=0), axis=1)

    # Plot the combined histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define bar width and positions
    bar_width = 0.3
    x = np.arange(len(combined_percentages))

    # Plot bars for each column
    ax.bar(
        x - bar_width / 2,
        combined_percentages["SNT_availability_2sc"],
        bar_width,
        label="Before S1B failure \n (two s/c available)",
        color="orange",
        alpha=0.7,
    )
    ax.bar(
        x + bar_width / 2,
        combined_percentages["SNT_availability_1sc"],
        bar_width,
        label="After S1B failure \n (one s/c available)",
        color="skyblue",
        alpha=0.7,
    )

    # Add labels and title
    ax.set_xlabel("Categories")
    ax.set_ylabel("Share of Bridges (%)")
    ax.set_title("Sentinel-1 Availability Over Bridges")

    ax.set_xticks(x)
    ax.set_xticklabels(combined_percentages.index)

    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Save the plot
    plt.savefig(
        os.path.join(plots_path, "combined_histogram_snt_availability.jpg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure to free up memory

    # See how availability changed

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
        "Lost 1 s/c": 1,
        "Increased time": 2,
        "No change": 3,
        "Availability lost completely": 4,
        "Increased s/c availability": 5,
    }

    # Create a new column with numerical values
    bridges_gdf["SNT_availability_change_num"] = bridges_gdf[
        "SNT_availability_change"
    ].map(snt_availability_change_mapping)

    # Define color dictionary for SNT_availability_change
    color_dict_snt_change = {
        1: "#fee090",
        2: "#fc8d59",
        3: "#91bfdb",
        4: "#d73027",
        5: "#4575b4",
    }

    # Define bounds and tick labels for SNT_availability_change
    bounds_snt_change = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    tick_labels_snt_change = [
        "Lost 1 s/c ({})".format(snt_change_counted["Lost 1 s/c"]),
        "Increased time ({})".format(snt_change_counted["Increased time"]),
        "No change ({})".format(snt_change_counted["No change"]),
        "Availability lost \n completely ({})".format(
            snt_change_counted["Availability lost completely"]
        ),
        "Increased s/c \n availability ({})".format(
            snt_change_counted["Increased s/c availability"]
        ),
    ]
    tick_locations_snt_change = [1, 2, 3, 4, 5]

    # Plot map for SNT_availability_change
    plot_map(
        bridges_gdf,
        "SNT_availability_change_num",
        "SNT Availability Change",
        color_dict_snt_change,
        bounds_snt_change,
        tick_labels_snt_change,
        tick_locations_snt_change,
        plots_path,
    )
    # ====================================================
    # Generate tables with regional summaries
    # ====================================================

    # Generate a table that shows the number of bridges in each monitoring class for each continent
    # First row should have the values for all bridges (globally)
    # Do it separately for shm and 2sc
    # Create a new column that assigns Monitoring class based on Monitoring_2sc
    create_class_column(
        df=bridges_gdf,
        source_col="Monitoring_2sc",
        target_col="Monitoring_2sc_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["No monitoring", "Low", "Medium", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    # Calculate the number of bridges in each monitoring class for each continent
    monitoring_results = calculate_monitoring_results(
        df=bridges_gdf,
        col_list=["Monitoring", "Monitoring_2sc_class"],
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

    # Save the table as a JPG image
    save_table_as_image(
        df=monitoring_results,
        title="Summary of monitoring capabilities by continent and category",
        file_path=os.path.join(plots_path, "table_monitoring_stats.jpg"),
        figsize=(10, 6),
        fontsize=10,
    )

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

    # Save the new table as a JPG image
    save_table_as_image(
        df=combined_monitoring_results,
        title="Summary of monitoring capabilities by continent and category",
        file_path=os.path.join(plots_path, "table_monitoring_stats_summary.jpg"),
        figsize=(10, 6),
        fontsize=10,
    )

    # Plot monitoring capabilities (SNT and SHM) by structural vulnerability

    # Define the order of vulnerabilities
    vulnerability_order = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Calculate the number of bridges in each monitoring class for each vulnerability level
    monitoring_results_vulnerability = calculate_vulnerability_monitoring_results(
        df=bridges_gdf,
        vulnerability_order=vulnerability_order,
        vulnerability_col="Vulnerability_norm",
        monitoring_cols=["Monitoring", "Monitoring_2sc_class"],
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

    # Save the table as a JPG image
    save_table_as_image(
        df=monitoring_results_vulnerability,
        title="Monitoring Capabilities by Structural Vulnerability",
        file_path=os.path.join(plots_path, "table_monitoring_vulnerability.jpg"),
        figsize=(12, 8),
        fontsize=10,
    )

    # ====================================================
    # Plot stacked histograms comparing SHM and SNT monitoring
    # ====================================================

    vulnerability_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    # vulnerability_order = [0.2, 0.4, 0.6, 0.7777, 1.0]
    bar_groups = [
        (["SHM 0", "SHM 1"], ["red", "green"]),
        (
            ["SNT No monitoring", "SNT Low", "SNT Medium", "SNT High", "SNT Very High"],
            ["red", "yellow", "orange", "lightgreen", "green"],
        ),
    ]

    plot_stacked_histogram(
        df=monitoring_results_vulnerability,
        x_labels=vulnerability_order,
        bar_groups=bar_groups,
        xlabel="Structural Vulnerability (Normalized)",
        ylabel="Number of Bridges",
        title="Number of Bridges \n by Structural Vulnerability and Monitoring Type",
        save_path=os.path.join(plots_path, "histogram_monitoring_vulnerability.jpg"),
    )

    # TO DO - do the same for bridge age, health

    # ====================================================
    # Plot table comparing SHM and SNT monitoring
    # ====================================================

    # Monitoring SHM vs SNT

    # Create a DataFrame to store the results
    monitoring_results = pd.DataFrame()

    # Calculate the number of bridges in each monitoring class
    shm_counts = bridges_gdf["Monitoring"].value_counts()
    snt_counts = bridges_gdf["Monitoring_2sc_class"].value_counts()

    # Create a cross-tabulation of SHM and SNT categories
    cross_tab = pd.crosstab(
        bridges_gdf["Monitoring"], bridges_gdf["Monitoring_2sc_class"]
    )

    # Reorder columns and rows
    cross_tab = cross_tab.reindex(
        columns=["No monitoring", "Low", "Medium", "High", "Very High"], fill_value=0
    )
    cross_tab = cross_tab.reindex(index=[0, 1], fill_value=0)

    # Save the table as an image
    save_table_as_image(
        df=cross_tab,
        title="Number of Bridges by SHM and SNT Categories",
        file_path=os.path.join(plots_path, "table_monitoring_shm_snt.jpg"),
        figsize=(12, 8),
        fontsize=10,
    )

    # ====================================================
    # Plot histogram of bridge parameters by PS availability
    # ====================================================

    # Plot PS availability by bridge type
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Type",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_type.jpg",
        fontsize=10,
    )

    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Type",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_type_count.jpg",
        fontsize=10,
        display_percentage=False,
    )

    # For each type of bridge, plot the histogram of PS availability
    # for each segment 1-5 so that for each bin, there are five bars

    segment_monitoring_col = [
        "Monitoring_1_PS_availability_only",
        "Monitoring_2_PS_availability_only",
        "Monitoring_3_PS_availability_only",
        "Monitoring_4_PS_availability_only",
        "Monitoring_5_PS_availability_only",
    ]

    new_column_names = {
        "Monitoring_1_PS_availability_only": "PS_avail_1_edge",
        "Monitoring_2_PS_availability_only": "PS_avail_2_intermediate",
        "Monitoring_3_PS_availability_only": "PS_avail_3_central",
        "Monitoring_4_PS_availability_only": "PS_avail_4_intermediate",
        "Monitoring_5_PS_availability_only": "PS_avail_5_edge",
    }

    # Rename columns in the DataFrame
    bridges_gdf.rename(columns=new_column_names, inplace=True)

    segment_monitoring_col_new = [
        "PS_avail_1_edge",
        "PS_avail_2_intermediate",
        "PS_avail_3_central",
        "PS_avail_4_intermediate",
        "PS_avail_5_edge",
    ]

    unique_types = bridges_gdf["Type"].unique()
    num_types = len(unique_types)
    num_cols = 3
    num_rows = (
        num_types + num_cols - 1
    ) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(18, 12), constrained_layout=True
    )
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, type in zip(axes, unique_types):
        gdf_type = bridges_gdf[bridges_gdf["Type"] == type][segment_monitoring_col_new]

        # Define colors for each column
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_monitoring_col_new)))

        weights = np.ones(len(gdf_type[segment_monitoring_col_new])) / len(
            gdf_type[segment_monitoring_col_new]
        )
        weights_proper_shape = np.tile(
            weights[:, np.newaxis], (1, len(segment_monitoring_col_new))
        )
        _, _, patches = ax.hist(
            gdf_type[segment_monitoring_col_new].to_numpy(),
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            rwidth=0.8,
            weights=weights_proper_shape,
            label=segment_monitoring_col_new,
            color=colors,
            alpha=0.7,
        )

        ax.set_title(f"{type} bridges")
        ax.set_xlabel("Value")
        ax.set_ylabel("Percentage of bridges")
        ax.set_ylim(0, 0.6)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    # Add the legend to the last subplot and position it outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(segment_monitoring_col_new),
    )

    # Remove any empty subplots
    for i in range(num_types, len(axes)):
        fig.delaxes(axes[i])

    # Add the main title
    fig.suptitle("Histograms of PS Availability by Segment \n", fontsize=25)

    # Save the plot
    plt.savefig(
        os.path.join(plots_path, "histogram_PS_avail_by_segment_by_type.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure to free up memory

    # Plot PS availability by bridge length
    # Categorise length
    bridges_gdf["Length_class"] = pd.cut(
        bridges_gdf["Length"],
        bins=[0, 500, 1000, 2000, 5000, 10000000],
        labels=[
            "Below 500 m",
            "[500 m, 1000 m)",
            "[1000 m, 2000 m)",
            "[2000 m, 5000 m)",
            "5000 m or more",
        ],
        right=False,
        include_lowest=True,
    )

    # Plot
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Length_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_length.jpg",
        bridge_parameters=[
            "Below 500 m",
            "[500 m, 1000 m)",
            "[1000 m, 2000 m)",
            "[2000 m, 5000 m)",
            "5000 m or more",
        ],
        fontsize=10,
    )

    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Length_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_length_count.jpg",
        bridge_parameters=[
            "Below 500 m",
            "[500 m, 1000 m)",
            "[1000 m, 2000 m)",
            "[2000 m, 5000 m)",
            "5000 m or more",
        ],
        fontsize=10,
        display_percentage=False,
    )

    # Plot PS availability by bridge azimuth
    # Categorise length
    bridges_gdf["Azimuth_class"] = pd.cut(
        bridges_gdf["azimuth_3"],
        bins=[0, 45, 135, 225, 315, 360],
        labels=[
            "0-45 deg",
            "45-135 deg",
            "135-225 deg",
            "225-315 deg",
            "315-360 deg",
        ],
        right=False,
        include_lowest=True,
    )

    # Plot
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Azimuth_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_azimuth.jpg",
        bridge_parameters=[
            "0-45 deg",
            "45-135 deg",
            "135-225 deg",
            "225-315 deg",
            "315-360 deg",
        ],
        fontsize=10,
    )

    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Azimuth_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_azimuth_count.jpg",
        bridge_parameters=[
            "0-45 deg",
            "45-135 deg",
            "135-225 deg",
            "225-315 deg",
            "315-360 deg",
        ],
        fontsize=10,
        display_percentage=False,
    )

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
        0: "N-S orientation (330-30 & 150-210)",
        1: "Angled orientation (30-60, 120-150, 210-240, 300-330)",
        2: "E-W orientation (60-120 & 240-300)",
    }

    # Map the numerical labels to the new category names
    bridges_gdf["Azimuth_numerical_class"] = bridges_gdf["Azimuth_numerical_class"].map(
        category_mapping
    )

    # Convert the new column to a categorical type with explicit categories
    bridges_gdf["Azimuth_numerical_class"] = pd.Categorical(
        bridges_gdf["Azimuth_numerical_class"],
        categories=[
            "N-S orientation (330-30 & 150-210)",
            "Angled orientation (30-60, 120-150, 210-240, 300-330)",
            "E-W orientation (60-120 & 240-300)",
        ],
        ordered=True,
    )

    # Plot
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Azimuth_numerical_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_azimuth_numerical.jpg",
        bridge_parameters=[
            "N-S orientation (330-30 & 150-210)",
            "Angled orientation (30-60, 120-150, 210-240, 300-330)",
            "E-W orientation (60-120 & 240-300)",
        ],
        fontsize=10,
    )

    # Plot
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Azimuth_numerical_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_azimuth_numerical_count.jpg",
        bridge_parameters=[
            "N-S orientation (330-30 & 150-210)",
            "Angled orientation (30-60, 120-150, 210-240, 300-330)",
            "E-W orientation (60-120 & 240-300)",
        ],
        fontsize=10,
        display_percentage=False,
    )

    # Plot PS availability by span length
    # Categorise span length
    bridges_gdf["Span_class"] = pd.cut(
        bridges_gdf["Maximum Span"],
        bins=[0, 300, 600, 900, 1200, 10000000],
        labels=[
            "Below 300 m",
            "[300 m, 600 m)",
            "[600 m, 900 m)",
            "[900 m, 1200 m)",
            "1200 m or more",
        ],
        right=False,
        include_lowest=True,
    )

    # Plot
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Span_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_span.jpg",
        bridge_parameters=[
            "Below 300 m",
            "[300 m, 600 m)",
            "[600 m, 900 m)",
            "[900 m, 1200 m)",
            "1200 m or more",
        ],
        fontsize=10,
    )

    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Span_class",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_bridge_span_count.jpg",
        bridge_parameters=[
            "Below 300 m",
            "[300 m, 600 m)",
            "[600 m, 900 m)",
            "[900 m, 1200 m)",
            "1200 m or more",
        ],
        fontsize=10,
        display_percentage=False,
    )

    # Plot by Deck Material
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Deck",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_deck_material.jpg",
        fontsize=10,
    )

    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Deck",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_deck_material_count.jpg",
        fontsize=10,
        display_percentage=False,
    )

    # Plot by Cable/Truss Material
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Cable/Truss",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_cable_material.jpg",
        fontsize=10,
    )
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Cable/Truss",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_cable_material_count.jpg",
        fontsize=10,
        display_percentage=False,
    )

    # Plot by Piers/Pylons Material
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Piers/Pylons",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_piers_material.jpg",
        fontsize=10,
    )
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material: Piers/Pylons",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_piers_material_count.jpg",
        fontsize=10,
        display_percentage=False,
    )

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

    # Plot by Material
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_material.jpg",
        fontsize=10,
    )
    plot_bridge_parameter_histogram(
        df=bridges_gdf,
        ps_column="Monitoring_PS_availability_only",
        parameter_column="Material",
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        plot_path=plots_path,
        plot_filename="histogram_ps_availability_by_material_count.jpg",
        fontsize=10,
        display_percentage=False,
    )

    # ====================================================
    # Plot scatter plots
    # ====================================================

    # Plot scatter plot of azimuth vs PS availability
    plt.figure(figsize=(12, 8))
    plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        s=10,
    )

    vertical_lines = [30, 60, 120, 150, 210, 240, 300, 330]
    for x in vertical_lines:
        plt.axvline(x=x, color="red", linestyle="--", linewidth=1)

    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    plt.savefig(
        os.path.join(plots_path, "scatter_azimuth_vs_ps_availability.jpg"),
        dpi=300,
        bbox_inches="tight",
    )

    # Plot scatter plot of azimuth vs PS availability by total length
    plt.figure(figsize=(12, 8))
    plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        c=bridges_gdf["Length"],
        cmap="viridis",
        s=10,
        vmax=5000,
    )
    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    plt.colorbar(label="Length")
    plt.savefig(
        os.path.join(plots_path, "scatter_azimuth_vs_ps_availability_length.jpg"),
        dpi=300,
        bbox_inches="tight",
    )

    # Plot scatter plot of azimuth vs PS availability by maximum span
    plt.figure(figsize=(12, 8))
    plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        c=bridges_gdf["Maximum Span"],
        cmap="viridis",
        s=10,
        vmax=1000,
    )
    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    plt.colorbar(label="Maximum Span")
    plt.savefig(
        os.path.join(plots_path, "scatter_azimuth_vs_ps_availability_span.jpg"),
        dpi=300,
        bbox_inches="tight",
    )

    # Factorize the 'Material: Deck' column to convert it to numerical values
    bridges_gdf["Material_Deck_numeric"], unique_materials = pd.factorize(
        bridges_gdf["Material: Deck"]
    )

    # Create a colormap with a unique color for each category
    num_categories = len(unique_materials)
    cmap = plt.cm.get_cmap(
        "tab20", num_categories
    )  # 'tab20' is a colormap with 20 distinct colors

    # Plot scatter plot of azimuth vs PS availability
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        c=bridges_gdf["Material_Deck_numeric"],
        cmap=cmap,
        s=10,
    )
    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    cbar = plt.colorbar(sc)
    cbar.set_label("Material: Deck (encoded)")
    cbar.set_ticks(range(num_categories))
    cbar.set_ticklabels(unique_materials)
    plt.savefig(
        os.path.join(
            plots_path, "scatter_azimuth_vs_ps_availability_material_deck.jpg"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # Factorize the region column to convert it to numerical values
    bridges_gdf["Region_numeric"], unique_materials = pd.factorize(
        bridges_gdf["Region"]
    )

    # Create a colormap with a unique color for each category
    num_categories = len(unique_materials)
    cmap = plt.cm.get_cmap(
        "tab20", num_categories
    )  # 'tab20' is a colormap with 20 distinct colors

    # Plot scatter plot of azimuth vs PS availability
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        c=bridges_gdf["Region_numeric"],
        cmap=cmap,
        s=10,
    )
    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    cbar = plt.colorbar(sc)
    cbar.set_label("Region_numeric")
    cbar.set_ticks(range(num_categories))
    cbar.set_ticklabels(unique_materials)
    plt.savefig(
        os.path.join(plots_path, "scatter_azimuth_vs_ps_availability_region.jpg"),
        dpi=300,
        bbox_inches="tight",
    )

    # Factorize the region column to convert it to numerical values
    bridges_gdf["Type_numeric"], unique_materials = pd.factorize(bridges_gdf["Type"])

    # Create a colormap with a unique color for each category
    num_categories = len(unique_materials)
    cmap = plt.cm.get_cmap(
        "tab20", num_categories
    )  # 'tab20' is a colormap with 20 distinct colors

    # Plot scatter plot of azimuth vs PS availability
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        bridges_gdf["azimuth_3"],
        bridges_gdf["Monitoring_PS_availability_only"],
        c=bridges_gdf["Type_numeric"],
        cmap=cmap,
        s=10,
    )
    plt.xlabel("Azimuth")
    plt.ylabel("PS Availability")
    plt.title("PS Availability by Azimuth")
    cbar = plt.colorbar(sc)
    cbar.set_label("Type_numeric")
    cbar.set_ticks(range(num_categories))
    cbar.set_ticklabels(unique_materials)
    plt.savefig(
        os.path.join(plots_path, "scatter_azimuth_vs_ps_availability_type.jpg"),
        dpi=300,
        bbox_inches="tight",
    )

    # ====================================================
    # Find strength of correlation between PS availability and other parameters
    # ====================================================

    # correlation_matrix = bridges_gdf[
    #     [
    #         "Type_numeric",
    #         "Length",
    #         "azimuth_3",
    #         "Azimuth_numerical_class",
    #         "Material_Deck_numeric",
    #         "Maximum Span",
    #         "Monitoring_PS_availability_only",
    #     ]
    # ].corr(
    #     method="pearson"
    # )  # You can replace 'pearson' with 'spearman' or 'kendall'

    # correlation_with_dependent = correlation_matrix[
    #     "Monitoring_PS_availability_only"
    # ].drop("Monitoring_PS_availability_only")
    # print(correlation_with_dependent)

    # ====================================================
    # Plot histogram of risks
    # ====================================================

    # Plot two histogram where on x will be the risk category and on y the number of bridges
    # that can be monitored. One plot should be for Multi-hazard_risk_shm and the other
    # for Multi-hazard_risk_2sc. For shm, the bridges that can be monitored are the ones
    # that have 1 in Monitoring column, and for 2sc the bar should be stacked and shows
    # the share of bridges that have Medium, High and Very High monitoring capabilities

    # Create a new column for Multi-hazard_risk_shm_class
    bridges_gdf["Multi-hazard_risk_shm_class"] = pd.cut(
        bridges_gdf["Multi-hazard_risk_shm"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    # Filter data for bridges that can be monitored (Monitoring == 1)
    monitored_bridges_shm = bridges_gdf[bridges_gdf["Monitoring"] == 1]

    # Create a new column for Multi-hazard_risk_2sc_class
    bridges_gdf["Multi-hazard_risk_2sc_class"] = pd.cut(
        bridges_gdf["Multi-hazard_risk_2sc"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    # Filter data for Multi-hazard_risk_2sc and categorize monitoring capabilities
    monitored_bridges_2sc = bridges_gdf[
        bridges_gdf["Monitoring_2sc_class"].isin(["Medium", "High", "Very High"])
    ]

    # Create a pivot table for stacked bar plot
    pivot_table = monitored_bridges_2sc.pivot_table(
        index="Multi-hazard_risk_2sc_class",
        columns="Monitoring_2sc_class",
        aggfunc="size",
        fill_value=0,
    )

    # Filter out columns with all zero values
    pivot_table = pivot_table.loc[:, (pivot_table != 0).any(axis=0)]

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))

    # Plot histogram for Multi-hazard_risk_shm_class
    monitored_bridges_shm[
        "Multi-hazard_risk_shm_class"
    ].value_counts().sort_index().plot(kind="bar", ax=axes[0])
    axes[0].set_xlabel("Risk Category")
    axes[0].set_ylabel("Number of Bridges")
    axes[0].set_title("Number of Bridges Monitored by SHM")
    axes[0].set_ylim(0, 180)

    # Plot stacked histogram for Multi-hazard_risk_2sc
    pivot_table.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_xlabel("Risk Category")
    axes[1].set_ylabel("Number of Bridges")
    axes[1].set_title("Number of Bridges Monitored by 2SC with Monitoring Capabilities")
    axes[1].legend(title="Monitoring Capability")
    axes[1].set_ylim(0, 180)

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_path, "histogram_multi_hazard_risk_by_monitoring_capabilities.jpg"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot the above on one plot
    # Prepare the data for SHM
    shm_counts = (
        monitored_bridges_shm["Multi-hazard_risk_shm_class"].value_counts().sort_index()
    )
    shm_counts.name = "SHM"

    # Prepare the data for 2SC
    pivot_table = bridges_gdf.pivot_table(
        index="Multi-hazard_risk_2sc_class",
        columns="Monitoring_2sc_class",
        aggfunc="size",
        fill_value=0,
    )

    # Combine the data into a single DataFrame
    combined_data = pd.concat([shm_counts, pivot_table], axis=1).fillna(0)

    # Rename columns by adding "SNT" at the beginning
    combined_data.rename(
        columns={
            "Very High": "SNT Very High",
            "No monitoring": "SNT No monitoring",
            "Low": "SNT Low",
            "Medium": "SNT Medium",
            "High": "SNT High",
        },
        inplace=True,
    )

    # Plot the combined data
    plot_stacked_histogram(
        df=combined_data,
        x_labels=combined_data.index.tolist(),
        bar_groups=[
            (["SHM"], ["blue"]),
            (
                [
                    "SNT Medium",
                    "SNT High",
                    "SNT Very High",
                ],
                ["orange", "lightgreen", "green"],
            ),
        ],
        xlabel="Risk Category",
        ylabel="Number of Bridges",
        title="Number of Bridges Monitored by SHM and 2SC",
        save_path=os.path.join(plots_path, "combined_histogram_multi_hazard_risk.jpg"),
    )

    # Use multiple histograms function to plot a histogram for Multi-hazard_risk_shm and Multi-hazard_risk_com
    # so that for each risk category there are two bars, one for SHM and one for combined monitoring

    # Create a new column for Multi-hazard_risk_com_class
    bridges_gdf["Multi-hazard_risk_com_class"] = pd.cut(
        bridges_gdf["Multi-hazard_risk_com"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    plot_multiple_histograms(
        bridges_gdf,
        np.arange(0, 1.1, 0.2).tolist(),
        ["Multi-hazard_risk_shm", "Multi-hazard_risk_com"],
        0.6,
        plots_path,
    )

    # Plot histogram that shows the monitoring on top of risk categories
    # Prepare the data for SHM
    create_class_column(
        df=bridges_gdf,
        source_col="Monitoring",
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
    )

    # Prepare the data for combined monitoring
    bridges_gdf["Monitoring_combined_class"] = bridges_gdf.apply(
        lambda row: (
            row["Monitoring_shm_class"]
            if row["Monitoring_shm_class"] == "SHM monitoring"
            else row["Monitoring_2sc_class"]
        ),
        axis=1,
    )

    pivot_table_combined_monitoring = bridges_gdf.pivot_table(
        index="Multi-hazard_risk_com_class",
        columns="Monitoring_combined_class",
        aggfunc="size",
        fill_value=0,
    )

    # Combine the data into a single DataFrame
    combined_data = pd.concat(
        [pivot_table_shm, pivot_table_combined_monitoring], axis=1
    ).fillna(0)

    # Remove duplicated column (SHM monitoring)
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    # Rename columns by adding "SNT" at the beginning
    combined_data.rename(
        columns={
            "Very High": "SNT Very High",
            "Low": "SNT Low",
            "Medium": "SNT Medium",
            "High": "SNT High",
        },
        inplace=True,
    )

    # Plot the combined data
    plot_stacked_histogram_new(
        df=combined_data,
        x_labels=combined_data.index.tolist(),
        bar_groups=[
            (["No SHM monitoring", "SHM monitoring"], ["red", "darkgreen"]),
            (
                [
                    "No monitoring",
                    "SNT Low",
                    "SNT Medium",
                    "SNT High",
                    "SNT Very High",
                    "SHM monitoring",
                ],
                ["red", "pink", "orange", "lightgreen", "green", "darkgreen"],
            ),
        ],
        xlabel="Risk Category",
        ylabel="Number of Bridges",
        title="Number of Bridges Monitored by SHM and 2SC",
        save_path=os.path.join(
            plots_path, "combined_histogram_multi_hazard_risk_and_monitoring.jpg"
        ),
    )


# ====================================================
# Plot tables for risk categories
# ====================================================

# Create a table that shows the number of bridges in each risk category for each continent
# First row should have the values for all bridges (globally)
# Do it separately for SHM and combined monitoring using columns
# Multi-hazard_risk_shm_class and Multi-hazard_risk_com_class


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

# Save the table as a JPG image
save_table_as_image(
    df=multi_hazard_shm_results,
    title="Summary of risk when only SHM is used",
    file_path=os.path.join(plots_path, "table_risk_shm_stats.jpg"),
    figsize=(10, 6),
    fontsize=10,
)

save_table_as_image(
    df=multi_hazard_com_results,
    title="Summary of risk when both SHM and SNT-1 are used",
    file_path=os.path.join(plots_path, "table_risk_com_stats.jpg"),
    figsize=(10, 6),
    fontsize=10,
)


print("end")
