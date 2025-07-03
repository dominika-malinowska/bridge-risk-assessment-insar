"""
This script can be used to create plots from the data in the 'Bridges' GeoDataFrame.
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from ps_predictions.second_paper.risk_calculation.utils.plotting_functions import (
    create_class_column,
    calculate_monitoring_results,
    save_table_as_image,
    calculate_vulnerability_monitoring_results,
)


def plot_ps_availability(
    bridges_gdf, column_name, desired_order, custom_cmap, plots_path, figure_id
):
    # Create a table counting the number of bridges by material and PS availability class
    table_PS_availability_by_material = pd.pivot_table(
        bridges_gdf,
        values="Monitoring_PS_availability_only",
        index=column_name,
        columns="Monitoring_PS_availability_classes",
        aggfunc="count",
        observed=False,
    )

    sum_for_each_category = table_PS_availability_by_material.sum(axis=1)

    # Normalize the counts to get the share for each category
    table_PS_availability_by_material = table_PS_availability_by_material.div(
        table_PS_availability_by_material.sum(axis=1), axis=0
    )

    # Reorder the index to the desired order
    table_PS_availability_by_material = table_PS_availability_by_material.reindex(
        desired_order[::-1]
    )
    sum_for_each_category = sum_for_each_category.reindex(desired_order[::-1])

    plt.figure(figsize=(7, 5))
    ax = table_PS_availability_by_material.plot(
        kind="barh", stacked=True, ax=plt.gca(), colormap=custom_cmap
    )
    plt.xlabel(
        "Share of the total number of bridges for that " + column_name.lower(),
        fontsize=18,
    )
    plt.xticks(
        plt.gca().get_xticks(),
        [f"{int(x * 100)}%" for x in plt.gca().get_xticks()],
        fontsize=16,
    )
    plt.xlim(0, 1)
    plt.ylabel(f"{column_name.capitalize()} (number of bridges)", fontsize=18)

    # y_ticks
    y_ticks = []
    for i in range(len(sum_for_each_category)):
        y_ticks.append(
            f"{sum_for_each_category.index[i]} \n ({sum_for_each_category.iloc[i]})"
        )
    plt.yticks(plt.gca().get_yticks(), y_ticks, fontsize=16)

    # plt.title(f"PS availability by {column_name.lower()}", fontsize=24)
    plt.legend(title="PS availability classes")

    # Plot the legend below the plot
    new_labels = ["Very Low", "Low", "Medium", "High", "Very high"]
    plt.legend(
        title="PS availability classes",
        labels=new_labels,
        bbox_to_anchor=(-0.3, -0.4),
        loc="lower left",
        ncol=5,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove the borders around the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.0f}%" for v in container.datavalues],
            label_type="center",
            fontsize=14,
            color="black",
        )

    plt.savefig(
        os.path.join(
            plots_path, f"{figure_id}_ps_availability_by_{column_name.lower()}.jpg"
        ),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def plot_ps_availability_segments(
    bridges_gdf, column_name, desired_order, custom_cmap, plots_path
):
    # Define the segments
    segments = [
        "PS_availability_edge",
        "PS_availability_intermediate",
        "PS_availability_central",
    ]

    # Create the Monitoring_PS_availability_classes column with range labels
    for segment in segments:
        bridges_gdf[f"{segment}_classes"] = pd.cut(
            bridges_gdf[segment],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"],
            right=True,
            include_lowest=True,
        )

    # Initialize a dictionary to hold the combined data
    combined_data = {segment: None for segment in segments}

    for segment in segments:
        # Create the pivot table with counts
        table_PS_availability_by_material = pd.pivot_table(
            bridges_gdf,
            values=segment,
            index=column_name,
            columns=f"{segment}_classes",
            aggfunc="count",
            observed=False,
        )

        # Normalize the counts to get the share for each category
        table_PS_availability_by_material = table_PS_availability_by_material.div(
            table_PS_availability_by_material.sum(axis=1), axis=0
        )

        # Reorder the index to the desired order
        table_PS_availability_by_material = table_PS_availability_by_material.reindex(
            desired_order
        )

        # Store the normalized data in the combined_data dictionary
        combined_data[segment] = table_PS_availability_by_material

    # Concatenate the data for the three segments
    combined_df = pd.concat(combined_data, axis=1)

    # Plot the combined data
    _, ax = plt.subplots(figsize=(7, 5))
    # Create a secondary y-axis
    ax2 = ax.twinx()

    bar_size = 0.2  # Reduced bar size to create space between bars
    padding = 0.5  # Padding between main categories
    inner_padding = 0.05  # Small padding between bars in each group
    y_locs = np.arange(len(desired_order)) * (bar_size * 3 + padding)

    colors = custom_cmap.colors
    for i, segment in enumerate(segments):
        segment_data = combined_df[segment]
        bottom = np.zeros(len(segment_data))
        for j, col in enumerate(segment_data.columns):
            ax.barh(
                y_locs + i * (bar_size + inner_padding),
                segment_data[col],
                align="edge",
                height=bar_size,
                color=colors[j],
                label=f"{segment} - {col}" if i == 0 else "",
                left=bottom,
            )
            bottom += segment_data[col]

    # Set labels and title
    ax.set_xlabel("Percent of total for that " + column_name.lower(), fontsize=18)
    ax.set_ylabel(column_name.capitalize(), fontsize=18)
    # ax.set_title(
    #     f"PS availability by {column_name.lower()} for different segments", fontsize=24
    # )
    ax.set_yticks(y_locs + bar_size * 1.5 + inner_padding)
    ax.set_yticklabels(desired_order, fontsize=16)

    # Set the secondary y-axis to match the primary y-axis
    ax2.set_ylim(ax.get_ylim())

    # Set the y-axis ticks and labels for the secondary y-axis
    secondary_labels = []
    for category in desired_order:
        secondary_labels.extend(
            [
                "Edge",
                "Intermediate",
                "Central",
            ]
        )

    # Calculate the y-tick positions for the secondary y-axis to match the primary y-axis
    secondary_y_ticks = np.sort(
        np.concatenate(
            [y_locs + (i + 0.5) * bar_size + i * inner_padding for i in range(3)]
        )
    )

    # Set the y-axis ticks and labels for the secondary y-axis
    ax2.set_yticks(secondary_y_ticks)
    ax2.set_yticklabels(secondary_labels)

    # Adjust the position of the secondary y-axis labels
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Set custom x-axis ticks with percentage signs
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.2)], fontsize=16
    )
    ax.set_xlim(0, 1)

    # Change the labels on the legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["No availability", "Low", "Medium", "High", "Very high"]
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(
        unique_handles,
        new_labels * 3,
        title="PS availability classes",
        bbox_to_anchor=(0.01, -0.27),
        loc="lower left",
        ncol=5,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove the borders around the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Remove the borders around the plot
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.0f}%" for v in container.datavalues],
            label_type="center",
            fontsize=14,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_path,
            f"histogram_ps_availability_by_segment_by_{column_name.lower()}.jpg",
        ),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def plot_ps_availability_segments_vertical(
    bridges_gdf, column_name, desired_order, custom_cmap, plots_path, figid
):
    # Define the segments
    segments = [
        "PS_availability_edge",
        "PS_availability_intermediate",
        "PS_availability_central",
    ]

    # Create the Monitoring_PS_availability_classes column with range labels
    for segment in segments:
        bridges_gdf[f"{segment}_classes"] = pd.cut(
            bridges_gdf[segment],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["[0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"],
            right=True,
            include_lowest=True,
        )

    # Initialize a dictionary to hold the combined data
    combined_data = {segment: None for segment in segments}

    for segment in segments:
        # Create the pivot table with counts
        table_PS_availability_by_material = pd.pivot_table(
            bridges_gdf,
            values=segment,
            index=column_name,
            columns=f"{segment}_classes",
            aggfunc="count",
            observed=False,
        )

        # Normalize the counts to get the share for each category
        table_PS_availability_by_material = table_PS_availability_by_material.div(
            table_PS_availability_by_material.sum(axis=1), axis=0
        )

        # Reorder the index to the desired order
        table_PS_availability_by_material = table_PS_availability_by_material.reindex(
            desired_order
        )

        # Store the normalized data in the combined_data dictionary
        combined_data[segment] = table_PS_availability_by_material

    # Concatenate the data for the three segments
    combined_df = pd.concat(combined_data, axis=1)

    # Plot the combined data
    _, ax = plt.subplots(figsize=(7, 14))
    # Create a secondary x-axis
    ax2 = ax.twiny()

    bar_width = 0.2  # Reduced bar width to create space between bars
    padding = 0.5  # Padding between main categories
    inner_padding = 0.05  # Small padding between bars in each group
    x_locs = np.arange(len(desired_order)) * (bar_width * 3 + padding)

    colors = custom_cmap.colors
    for i, segment in enumerate(segments):
        segment_data = combined_df[segment]
        bottom = np.zeros(len(segment_data))
        for j, col in enumerate(segment_data.columns):
            ax.bar(
                x_locs + i * (bar_width + inner_padding),
                segment_data[col],
                align="edge",
                width=bar_width,
                color=colors[j],
                label=f"{segment} - {col}" if i == 0 else "",
                bottom=bottom,
            )
            bottom += segment_data[col]

    # Set labels and title
    ax.set_ylabel("Share of total for given \n bridge type & segment", fontsize=18)
    ax.set_xlabel(column_name.capitalize(), fontsize=18)
    # ax.set_title(
    #     f"PS availability by {column_name.lower()} for different segments", fontsize=24
    # )
    ax.set_xticks(x_locs + bar_width * 1.5 + inner_padding)
    ax.set_xticklabels(desired_order, fontsize=16)
    ax.tick_params(axis="x", labelsize=16, rotation=30)

    # Set the secondary x-axis to match the primary x-axis
    ax2.set_xlim(ax.get_xlim())

    # Set the x-axis ticks and labels for the secondary x-axis
    secondary_labels = []
    for category in desired_order:
        secondary_labels.extend(
            [
                "E",
                "I",
                "C",
            ]
        )

    # Calculate the x-tick positions for the secondary x-axis to match the primary x-axis
    secondary_x_ticks = np.sort(
        np.concatenate(
            [x_locs + (i + 0.5) * bar_width + i * inner_padding for i in range(3)]
        )
    )

    # Set the x-axis ticks and labels for the secondary x-axis
    ax2.set_xticks(secondary_x_ticks)
    ax2.set_xticklabels(secondary_labels, fontsize=16)
    ax2.tick_params(axis="x", labelsize=16)

    # Move the original x-axis to the bottom but slightly above the secondary axis
    ax.spines["bottom"].set_position(("outward", 25))
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Move the secondary x-axis to the bottom
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.tick_bottom()

    ax.tick_params(bottom=False)
    ax2.tick_params(bottom=True)

    # Set custom y-axis ticks with percentage signs
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.2)],
        fontsize=16,
    )
    ax.set_ylim(0, 1)

    # Change the labels on the legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["No availability", "Low", "Medium", "High", "Very high"]
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(
        unique_handles,
        new_labels * 3,
        title="PS availability classes",
        bbox_to_anchor=(-0.2, -0.97),
        loc="lower left",
        ncol=3,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove the borders around the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Remove the borders around the plot
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    # for container in ax.containers:
    #     ax.bar_label(
    #         container,
    #         labels=[f"{v * 100:.0f}%" for v in container.datavalues],
    #         label_type="center",
    #         fontsize=11,
    #         color="black",
    #     )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_path,
            f"{figid}_ps_availability_by_segment_by_{column_name.lower()}_vertical.jpg",
        ),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
    # Save raw data
    bridges_gdf[
        [
            "ID",
            "PS_availability_edge",
            "PS_availability_intermediate",
            "PS_availability_central",
            "PS_availability_edge_classes",
            "PS_availability_intermediate_classes",
            "PS_availability_central_classes",
            column_name,
        ]
    ].to_csv(
        os.path.join(
            plots_path,
            f"{figid}_ps_availability_by_segment_by_{column_name.lower()}.csv",
        ),
        index=False,
    )


def plotting_stacked_histogram(
    df,
    y_max,
    hatch_pattern,
    x_label,
    custom_handles,
    ncol,
    legend_title,
    file_title,
    plots_path,
    x_cat_order,
    bar_groups,
    custom_cmap,
):
    # # Calculate the required width for the legend
    # legend_width_per_col = 2.5  # Width per column in inches (adjust as needed)
    # legend_width = ncol * legend_width_per_col

    # Set the figure width based on the legend width
    # fig_width = max(12, legend_width)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Define bar width and spacing
    bar_width = 0.3
    bar_spacing = 0.1

    # Define x-axis labels and positions
    x_labels = x_cat_order
    x_positions = np.arange(len(x_labels))

    # Add grey horizontal lines every 50 units
    ax.set_yticks(np.arange(0, y_max + 50, 50))
    ax.tick_params(axis="y", labelsize=16)
    ax.yaxis.grid(True, which="both", color="grey", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)  # Ensure grid lines appear behind the plot

    # Plot each bar group
    for i, (categories, colors) in enumerate(bar_groups):
        bottom = np.zeros(len(x_labels))
        # for category, color in zip(categories, colors):
        for j, (category, color) in enumerate(zip(categories, colors)):

            hatch = hatch_pattern if "SHM" in category else ""
            # Determine edge color
            if i == 0:
                edgecolor = "lightgrey"  # Leftmost bar

            else:
                edgecolor = "black"  # Other bars

            _ = ax.bar(
                x_positions
                + (i * (bar_width + bar_spacing))
                - ((len(bar_groups) - 1) * (bar_width + bar_spacing)) / 2,
                df[category],
                bar_width,
                bottom=bottom,
                label=category,
                color=color,
                hatch=hatch,
                edgecolor=edgecolor,
                linewidth=0.4,  # Adjust the line width as needed
                zorder=3,  # Ensure bars are drawn above the grid lines
            )
            bottom += df[category]

    # Set axis labels and ticks
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel("Number of Bridges", fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=16)

    # Plot the legend below the plot
    plt.legend(
        title=legend_title,
        handles=custom_handles,
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=ncol,
        fontsize=16,
        columnspacing=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Save the plot
    plt.savefig(
        os.path.join(plots_path, file_title),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":

    # ====================================================
    # Define the paths and read the data
    # ====================================================

    # Define the path to the data
    data_path = "/mnt/g/RISK_PAPER"

    plots_path = os.path.join(data_path, "plots_new_data_02062025")

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # # Read the bridges gdf from a pickle
    bridges_gdf = pd.read_pickle(
        os.path.join(data_path, "lsb_risk_analysis_filtered.pkl")
        # os.path.join(data_path, "lsb_risk_analysis.pkl")
    )

    # Remove bridge with ID 755 that is incorrect due to lack of OSM lines
    bridges_gdf = bridges_gdf[bridges_gdf["ID"] != 755]
    # Reset the index
    bridges_gdf.reset_index(drop=True, inplace=True)

    # Print total number of bridges
    print(f"Total number of bridges: {len(bridges_gdf)}")

    # Set the font size
    plt.rcParams["font.size"] = 18

    # Cmap for the PS availabilty (from no availibility to great availability)
    # custom_cmap = ListedColormap(
    #     ["#0868ac", "#43a2ca", "#7bccc4", "#bae4bc", "#f0f9e8"]
    # )
    alpha = 0.9  # Set the desired alpha value

    custom_cmap = ListedColormap(
        [
            to_rgba("#CC3311", alpha),
            to_rgba("#EE7733", alpha),
            to_rgba("#009988", alpha),
            to_rgba("#33BBEE", alpha),
            to_rgba("#0077BB", alpha),
        ]
    )
    custom_cmap_reversed = ListedColormap(custom_cmap.colors[::-1])

    # Create a asymmetrical cmap for heatmap
    # Define the list of colors with alpha
    colors = [
        to_rgba("#0077BB", alpha),
        to_rgba("#33BBEE", alpha),
        to_rgba("#009988", alpha),
        to_rgba("#EE7733", alpha),
        to_rgba("#CC3311", alpha),
    ]

    # Create a continuous colormap
    custom_cmap_asymmetrical = LinearSegmentedColormap.from_list(
        "custom_cmap_continuous", colors
    )

    # Create a symmetrical cmap for heatmap
    # Define the list of colors with alpha
    colors = [
        to_rgba("#FFAABB", alpha),
        to_rgba("#EE8866", alpha),
        to_rgba("#44BB99", alpha),
        to_rgba("#99DDFF", alpha),
        to_rgba("#77AADD", alpha),
        to_rgba("#99DDFF", alpha),
        to_rgba("#44BB99", alpha),
        to_rgba("#EE8866", alpha),
        to_rgba("#FFAABB", alpha),
    ]

    # Create a continuous colormap
    custom_cmap_symmetric = LinearSegmentedColormap.from_list(
        "custom_cmap_continuous", colors
    )

    # Define a color palette for the categories
    color_palette_SNT = {
        "6 days\n 2 s/c": to_rgba("#0077BB", alpha),
        "12 days\n 2 s/c": to_rgba("#33BBEE", alpha),
        "6 days\n 1 s/c": to_rgba("#009988", alpha),
        "12 days\n 1 s/c": to_rgba("#EE7733", alpha),
        "No SNT \n availability": to_rgba("#CC3311", alpha),
    }

    # custom_cmap = ListedColormap(
    #     ["#882255", "#88CCEE", "#117733", "#DDCC77", "#332288"]
    # )
    # custom_cmap = "viridis"
    # custom_cmap = "cividis"
    # "tableau-colorblind10"
    # custom_cmap = get_cmap("tab10")

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

    # Plot histogram of PS availability classes
    fig, ax = plt.subplots(figsize=(7, 5))
    bridges_gdf["Monitoring_PS_availability_classes"].value_counts(
        normalize=True
    ).sort_index().plot(
        kind="bar",
        color=[
            to_rgba("#CC3311", alpha),
            to_rgba("#EE7733", alpha),
            to_rgba("#009988", alpha),
            to_rgba("#33BBEE", alpha),
            to_rgba("#0077BB", alpha),
        ],
        ax=ax,
        zorder=3,  # Ensure bars are drawn above the grid lines
        # edgecolor="black",
        # linewidth=1,
    )

    ax.yaxis.grid(True, color="grey", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)  # Ensure grid lines appear behind the plot

    # Add values at the top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=14,
        )

    # Convert y-axis to percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Customize plot
    plt.xlabel("PS Availability Classes", fontsize=18)
    plt.xticks(
        ticks=range(5),
        labels=["Very Low", "Low", "Medium", "High", "Very high"],
        rotation=0,
        fontsize=16,
    )
    plt.ylabel("Share of Bridges", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)  # Adjust the font size for the y-tick labels
    plt.ylim(0, 0.4)

    # plt.title("Histogram of PS Availability Classes")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig1a_ps_availability.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Save raw data
    bridges_gdf[
        ["ID", "Monitoring_PS_availability_only", "Monitoring_PS_availability_classes"]
    ].to_csv(os.path.join(plots_path, "fig1a_ps_availability.csv"), index=False)

    # ====================================================
    # Plot SNT availability
    # ====================================================

    # Map the labels to the corresponding colors
    # color_map = [
    #     color_palette_SNT["6 days\n 2 s/c"],  # 2
    #     color_palette_SNT["12 days\n 2 s/c"],  # 4
    #     color_palette_SNT["6 days\n 1 s/c"],  # 1
    #     color_palette_SNT["12 days\n 1 s/c"],  # 3
    # ]

    color_map = [
        color_palette_SNT["12 days\n 1 s/c"],  # 3
        color_palette_SNT["6 days\n 1 s/c"],  # 1
        color_palette_SNT["12 days\n 2 s/c"],  # 4
        color_palette_SNT["6 days\n 2 s/c"],  # 2
    ]

    # Plot histogram of PS availability classes
    fig, ax = plt.subplots(figsize=(7, 5))

    bridges_gdf["SNT_availability_2sc"].value_counts(
        normalize=True
        # ).sort_index().reindex([2, 4, 1, 3]).plot(
    ).sort_index().reindex([3, 1, 4, 2]).plot(
        kind="bar",
        color=color_map,
        ax=ax,
        zorder=3,  # Ensure bars are drawn above the grid lines
        # edgecolor="black",
        # linewidth=1,
    )

    ax.yaxis.grid(True, color="grey", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)  # Ensure grid lines appear behind the plot

    # Add values at the top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=14,
        )

    # Convert y-axis to percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Customize plot
    plt.xlabel("Sentinel-1 availability", fontsize=18)
    plt.xticks(
        ticks=range(4),
        labels=[
            # "6 days\n 2 s/c",
            # "12 days\n 2 s/c",
            # "6 days\n 1 s/c",
            # "12 days\n 1 s/c",
            "12 days\n 1 s/c",
            "6 days\n 1 s/c",
            "12 days\n 2 s/c",
            "6 days\n 2 s/c",
        ],
        rotation=0,
        fontsize=16,
    )
    plt.ylabel("Share of Bridges", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    plt.ylim(0, 0.4)

    # plt.title("Histogram of Sentinel-1 availability")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig1b_snt_avail.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()
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
    ].to_csv(os.path.join(plots_path, "fig1b_snt_avail.csv"), index=False)

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

    # Plot histogram of PS availability classes
    fig, ax = plt.subplots(figsize=(7, 5))

    bridges_gdf["Spaceborne monitoring class"].value_counts(
        normalize=True
    ).sort_index().plot(
        kind="bar",
        color=[
            to_rgba("#CC3311", alpha),
            to_rgba("#EE7733", alpha),
            to_rgba("#009988", alpha),
            to_rgba("#33BBEE", alpha),
            to_rgba("#0077BB", alpha),
        ],
        ax=ax,
        zorder=3,  # Ensure bars are drawn above the grid lines
        # edgecolor="black",
        # linewidth=1,
    )

    ax.yaxis.grid(True, color="grey", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)  # Ensure grid lines appear behind the plot

    # Add values at the top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=14,
        )

    # Convert y-axis to percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Customize plot
    plt.xlabel("Spaceborne Monitoring Classes", fontsize=18)
    plt.xticks(
        ticks=range(5),
        labels=["No monitoring", "Low", "Medium", "High", "Very high"],
        rotation=0,
        fontsize=16,
    )
    plt.ylabel("Share of Bridges", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)  # Adjust the font size for the y-tick labels
    plt.ylim(0, 0.4)

    # plt.title("Histogram of spaceborne monitoring")

    # Remove the borders around the plot
    # ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add horizontal brackets to group categories
    left_bracket_x = [0, 1]  # Positions for "No monitoring" and "Low"
    right_bracket_x = [2, 3, 4]  # Positions for "Medium", "High", and "Very high"

    # Get the height to place the brackets
    bracket_height = 0.25  # Overlap with the plot
    text_height = bracket_height + 0.02  # Position for the text

    # Draw the "Monitoring unlikely" bracket
    ax.plot(
        [left_bracket_x[0] - 0.25, left_bracket_x[-1] + 0.25],
        [bracket_height, bracket_height],
        "k-",
        linewidth=1.5,
    )
    # Add vertical ends to the bracket
    ax.plot(
        [left_bracket_x[0] - 0.25, left_bracket_x[0] - 0.25],
        [bracket_height, bracket_height - 0.01],
        "k-",
        linewidth=1.5,
    )
    ax.plot(
        [left_bracket_x[-1] + 0.25, left_bracket_x[-1] + 0.25],
        [bracket_height, bracket_height - 0.01],
        "k-",
        linewidth=1.5,
    )
    # Add text
    ax.text(
        (left_bracket_x[0] + left_bracket_x[-1]) / 2,
        text_height,
        "Spaceborne\nmonitoring unlikely",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    # Draw the "Monitoring possible" bracket
    ax.plot(
        [right_bracket_x[0] - 0.25, right_bracket_x[-1] + 0.25],
        [bracket_height, bracket_height],
        "k-",
        linewidth=1.5,
    )
    # Add vertical ends to the bracket
    ax.plot(
        [right_bracket_x[0] - 0.25, right_bracket_x[0] - 0.25],
        [bracket_height, bracket_height - 0.01],
        "k-",
        linewidth=1.5,
    )
    ax.plot(
        [right_bracket_x[-1] + 0.25, right_bracket_x[-1] + 0.25],
        [bracket_height, bracket_height - 0.01],
        "k-",
        linewidth=1.5,
    )
    # Add text
    ax.text(
        (right_bracket_x[0] + right_bracket_x[-1]) / 2,
        text_height,
        "Spaceborne\nmonitoring possible",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    # Update y-axis limit to accommodate the brackets and text
    # plt.ylim(0, 0.45)

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig1c_space_monitoring.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Save raw data
    bridges_gdf[["ID", "Monitoring_2sc", "Spaceborne monitoring class"]].to_csv(
        os.path.join(plots_path, "fig1c_space_monitoring.csv"), index=False
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

    desired_order = ["Steel", "Concrete", "Composite", "Other"]
    plot_ps_availability(
        bridges_gdf, "Material", desired_order, custom_cmap, plots_path, "fig2a"
    )

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
        os.path.join(plots_path, "fig2a_ps_availability_by_material.csv"),
        index=False,
    )

    # ========== Plot by bridge type ==========
    bridges_gdf.rename(columns={"Type": "Bridge Type"}, inplace=True)

    bridges_gdf["Bridge Type"] = bridges_gdf["Bridge Type"].replace(
        "Cable Stayed", "Cable-Stayed"
    )

    desired_order = ["Arch", "Cable-Stayed", "Cantilever", "Truss", "Suspension"]
    plot_ps_availability(
        bridges_gdf, "Bridge Type", desired_order, custom_cmap, plots_path, "fig2b"
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
        os.path.join(plots_path, "fig2b_ps_availability_by_bridge_type.csv"),
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

    # Plot
    desired_order = ["Arch", "Cable-Stayed", "Cantilever", "Truss", "Suspension"]
    plot_ps_availability_segments_vertical(
        bridges_gdf, "Bridge Type", desired_order, custom_cmap, plots_path, "figSup1"
    )

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
            plots_path, "figSup1_ps_availability_by_segment_and_bridge_type.csv"
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

    desired_order = [
        "N-S orientation",
        "Angled orientation",
        "E-W orientation",
    ]
    plot_ps_availability(
        bridges_gdf, "Azimuth", desired_order, custom_cmap, plots_path, "fig2c"
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
        os.path.join(plots_path, "fig2c_ps_availability_by_azimuth.csv"),
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

    # Save the table as an image
    save_table_as_image(
        df=cross_tab,
        title="Number of Bridges by SHM and SNT Categories",
        file_path=os.path.join(plots_path, "table_monitoring_shm_snt.jpg"),
        figsize=(12, 8),
        fontsize=10,
    )

    # Define color dictionary
    color_dict_monitoring_change = {
        "No monitoring": "#CC3311",
        "Low": "#EE7733",
        "Medium": "#009988",
        "High": "#33BBEE",
        "Very High": "#0077BB",
    }

    # Plotting stacked horizontal bar chart with counts
    plt.figure(figsize=(7, 8))
    ax = cross_tab.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        color=[
            color_dict_monitoring_change[category] for category in cross_tab.columns
        ],
    )

    plt.ylabel("Number of Bridges", fontsize=18)
    plt.xlabel("SHM-based Monitoring", fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16, rotation=0)

    # Adjust legend to be on the right side
    plt.legend(
        title="Spaceborne Monitoring Class",
        bbox_to_anchor=(-0.15, -0.35),
        loc="lower left",
        ncol=3,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove the top, right, and bottom spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add the count as a number in the middle of each stacked bar segment
    for container in ax.containers:
        labels = [int(v.get_height()) if v.get_height() > 0 else "" for v in container]
        ax.bar_label(
            container, labels=labels, label_type="center", fontsize=14, color="black"
        )

    # Add vertical bracket to show sum of No SHM Medium, High and Very High

    # Calculate proper bracket dimensions to cover all three segments
    bracket_bottom = ax.patches[4].get_y()
    bracket_top = ax.patches[8].get_y() + ax.patches[8].get_height()
    bracket_height = bracket_top - bracket_bottom

    # Draw the vertical bracket on the left side of the No SHM bar
    bracket_x_pos = 0.35
    # Main vertical line
    ax.plot(
        [bracket_x_pos, bracket_x_pos],
        [bracket_bottom, bracket_bottom + bracket_height],
        "k-",
        linewidth=1.5,
    )
    # Add horizontal ends to the bracket
    ax.plot(
        [bracket_x_pos, bracket_x_pos - 0.02],
        [bracket_bottom, bracket_bottom],
        "k-",
        linewidth=1.5,
    )
    ax.plot(
        [bracket_x_pos, bracket_x_pos - 0.02],
        [bracket_bottom + bracket_height, bracket_bottom + bracket_height],
        "k-",
        linewidth=1.5,
    )

    # Add text label
    ax.text(
        bracket_x_pos + 0.15,
        bracket_bottom + bracket_height / 2,
        f"{int(ax.patches[4].get_height() + ax.patches[6].get_height() + ax.patches[8].get_height())}",
        ha="right",
        va="center",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig3b_SHM_and_space_monitoring_overlap.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

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
        os.path.join(plots_path, "fig3b_SHM_and_space_monitoring_overlap.csv"),
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

    # Save the new table as a JPG image
    save_table_as_image(
        df=combined_monitoring_results,
        title="Summary of monitoring capabilities by continent and category",
        file_path=os.path.join(plots_path, "table_monitoring_stats_summary.jpg"),
        figsize=(10, 6),
        fontsize=10,
    )

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

    # Plot the monitoring results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 10))

    # Plot SHM monitoring by continent
    shm_monitoring_by_continent[["No_SHM", "SHM"]].plot(
        kind="barh", stacked=True, ax=ax1, colormap=custom_cmap
    )
    ax1.set_xlabel("Share of the total \nnumber of bridges", fontsize=18)
    ax1.set_ylabel("Region (total number of bridges)", fontsize=18)
    ax1.set_xlim(0, 1)
    ax1.set_title("SHM \nMonitoring", fontsize=20)
    ax1.legend(
        title="Monitoring \navailability \nclasses",
        labels=["No SHM \nsensors", "SHM sensors \navailable"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        fontsize=16,
        columnspacing=0.8,
    )

    # y_ticks
    y_ticks = []
    for i in range(len(shm_monitoring_by_continent)):
        # index_label = "\n".join(shm_monitoring_by_continent.index[i].split())
        index_label = shm_monitoring_by_continent.index[i]
        if index_label == "Latin America/Caribbean":
            index_label = "Latin America \n/Caribbean"
            y_ticks.append(
                f"{index_label} ({shm_monitoring_by_continent['Total count of bridges'].iloc[i]})"
            )
        else:
            y_ticks.append(
                f"{index_label} \n ({shm_monitoring_by_continent['Total count of bridges'].iloc[i]})"
            )
    ax1.set_yticks(ax1.get_yticks())  # Ensure the y-ticks are set correctly
    ax1.set_yticklabels(y_ticks, fontsize=16)

    # Reverse the order on the y-axis
    ax1.invert_yaxis()

    # Plot spaceborne monitoring by continent
    space_monitoring_by_continent.plot(
        kind="barh", stacked=True, ax=ax2, colormap=custom_cmap
    )
    ax2.set_xlabel("Share of the total \nnumber of bridges", fontsize=18)
    ax2.set_xlim(0, 1)
    ax2.set_title("Spaceborne \nMonitoring", fontsize=20)
    ax2.legend(
        title="Monitoring \navailability \nclasses",
        labels=[
            "Spaceborne \nmonitoring \nunlikely",
            "Spaceborne \nmonitoring \npossible",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove y-ticks for ax2
    ax2.set_yticks([])

    # Reverse the order on the y-axis
    ax2.invert_yaxis()

    # Set custom x-axis ticks with percentage signs for both subplots
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, 1.1, 0.5))
        ax.set_xticklabels(
            [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.5)], fontsize=16
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Add the share as a number in the middle of each stacked bar
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[
                    f"{v * 100:.0f}%" if v > 0 else "" for v in container.datavalues
                ],
                label_type="center",
                fontsize=14,
                color="black",
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig3a_monitoring_by_continent.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

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
    ].to_csv(os.path.join(plots_path, "fig3a_monitoring_by_continent.csv"), index=False)

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

    # Define a custom color palette for the categories
    color_palette = {
        "Very Low": custom_cmap_reversed(0),
        "Low": custom_cmap_reversed(1 / 4),
        "Medium": custom_cmap_reversed(2 / 4),
        "High": custom_cmap_reversed(3 / 4),
        "Very High": custom_cmap_reversed(4 / 4),
    }

    # Extract the colors in the order of the categories
    colors = [
        color_palette[category]
        for category in ["Very Low", "Low", "Medium", "High", "Very High"]
    ]

    # Plot the monitoring results
    fig, ax = plt.subplots(figsize=(7, 16))

    # Plot SHM monitoring by continent
    vulnerability_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ].plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_ylabel("Share of the total number \nof bridges", fontsize=18)
    ax.set_xlabel("Region (total number of bridges)", fontsize=18)
    ax.set_ylim(0, 1)
    # ax.set_title("Structural vulnerability distribution by continent")
    ax.legend(
        title="Vulnerability classes",
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.97),
        ncol=3,
        fontsize=16,
        columnspacing=0.8,
    )

    x_ticks = []
    for i in range(len(vulnerability_results)):
        index_label = vulnerability_results.index[i]
        x_ticks.append(
            f"{index_label} \n ({vulnerability_results['Total count of bridges'].iloc[i]})"
        )
    ax.set_xticks(ax.get_xticks())  # Ensure the y-ticks are set correctly
    ax.set_xticklabels(x_ticks, fontsize=16, rotation=90)

    # Set custom y-axis ticks with percentage signs for both subplots
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.2)], fontsize=16
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.0f}%" if v > 0 else "" for v in container.datavalues],
            label_type="center",
            fontsize=12,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig4a_vulnerability_by_continent.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Vulnerability_norm",
            "Vulnerability_norm_class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_path, "fig4a_vulnerability_by_continent.csv"),
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

    # Save the table as a JPG image
    save_table_as_image(
        df=monitoring_results_vulnerability,
        title="Monitoring Capabilities by Structural Vulnerability",
        file_path=os.path.join(plots_path, "table_monitoring_vulnerability.jpg"),
        figsize=(12, 8),
        fontsize=10,
    )

    vulnerability_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    bar_groups = [
        (["SHM 0", "SHM 1"], [custom_cmap(0), custom_cmap(4)]),
        (
            ["SNT No monitoring", "SNT Low", "SNT Medium", "SNT High", "SNT Very High"],
            [custom_cmap(i / 4) for i in range(5)],
        ),
    ]
    custom_handles = [
        mpatches.Patch(facecolor="white", label="SHM:"),
        mpatches.Patch(facecolor="white", label="Spaceborne \nmonitoring:"),
        mpatches.Patch(facecolor="white", label=" "),
        mpatches.Patch(facecolor="white", label=" "),
        mpatches.Patch(
            facecolor=custom_cmap(0),
            hatch="..",
            label="No availability",
            edgecolor="lightgrey",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(0),
            label="No availability",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(2 / 4),
            label="Medium",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(4 / 4),
            label="Very high",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(4),
            hatch="..",
            label="Available",
            edgecolor="lightgrey",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(1),
            label="Low",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(3 / 4),
            label="High",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(facecolor="white", label=" "),
        # mpatches.Patch(facecolor="white", label=" "),
    ]

    plotting_stacked_histogram(
        df=monitoring_results_vulnerability,
        y_max=250,
        hatch_pattern="..",
        x_label="Structural Vulnerability",
        custom_handles=custom_handles,
        ncol=3,
        legend_title="Monitoring availability",
        file_title="fig4b_monitoring_vulnerability.jpg",
        plots_path=plots_path,
        x_cat_order=vulnerability_order,
        bar_groups=bar_groups,
        custom_cmap=custom_cmap,
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
        os.path.join(plots_path, "fig4b_monitoring_vulnerability.csv"),
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

    # Define a custom color palette for the categories
    color_palette = {
        "Very Low": custom_cmap_reversed(0),
        "Low": custom_cmap_reversed(1 / 4),
        "Medium": custom_cmap_reversed(2 / 4),
        "High": custom_cmap_reversed(3 / 4),
        "Very High": custom_cmap_reversed(4 / 4),
    }

    # Extract the colors in the order of the categories
    colors = [
        color_palette[category]
        for category in ["Very Low", "Low", "Medium", "High", "Very High"]
    ]

    # Plot the monitoring results
    fig, ax = plt.subplots(figsize=(7, 16))

    # Plot SHM monitoring by continent
    risk_results[
        [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ]
    ].plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_ylabel("Share of the total \nnumber of bridges", fontsize=18)
    ax.set_xlabel("Region (total number of bridges)", fontsize=18)
    ax.set_ylim(0, 1)
    # ax.set_title("Risk distribution by continent")
    ax.legend(
        title="Risk classes",
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.97),
        ncol=3,
        fontsize=16,
        columnspacing=0.8,
    )

    continent_labels = []
    for i in range(len(risk_results)):
        index_label = risk_results.index[i]
        continent_labels.append(
            f"{index_label} \n ({risk_results['Total count of bridges'].iloc[i]})"
        )
    ax.set_xticks(ax.get_xticks())  # Ensure the y-ticks are set correctly
    ax.set_xticklabels(continent_labels, fontsize=16, rotation=90)

    # Set custom y-axis ticks with percentage signs for both subplots
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(
        [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.2)], fontsize=16
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.0f}%" if v > 0 else "" for v in container.datavalues],
            label_type="center",
            fontsize=12,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig5a_risk_by_continent.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    # Save raw data
    bridges_gdf[
        [
            "ID",
            "Multi-hazard_risk_shm",
            "Multi-hazard_risk_shm_class",
            "Region",
        ]
    ].to_csv(
        os.path.join(plots_path, "fig5a_risk_by_continent.csv"),
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

    risk_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    colors_combined_monitoring = [custom_cmap(i / 4) for i in range(5)]
    colors_combined_monitoring.append(custom_cmap(4))
    bar_groups = [
        (["No SHM monitoring", "SHM monitoring"], [custom_cmap(0), custom_cmap(4)]),
        (
            [
                "No monitoring",
                "SNT Low",
                "SNT Medium",
                "SNT High",
                "SNT Very High",
                "SHM monitoring",
            ],
            colors_combined_monitoring,
        ),
    ]
    custom_handles = [
        mpatches.Patch(facecolor="white", label="SHM:"),
        mpatches.Patch(facecolor="white", label="Combined \nmonitoring:"),
        mpatches.Patch(facecolor="white", label=" "),
        mpatches.Patch(facecolor="white", label=" "),
        mpatches.Patch(
            facecolor=custom_cmap(0),
            hatch="..",
            label="No availability",
            edgecolor="lightgrey",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(0),
            label="No availability",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(2 / 4),
            label="SNT Medium",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(4 / 4),
            label="SNT Very high",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(4),
            hatch="..",
            label="Available",
            edgecolor="lightgrey",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(1), label="SNT Low", edgecolor="black", linewidth=1
        ),
        mpatches.Patch(
            facecolor=custom_cmap(3 / 4),
            label="SNT High",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Patch(
            facecolor=custom_cmap(4),
            hatch="..",
            label="SHM monitoring",
            edgecolor="black",
            linewidth=1,
        ),
    ]

    plotting_stacked_histogram(
        df=combined_risk,
        y_max=400,
        hatch_pattern="..",
        x_label="Risk",
        custom_handles=custom_handles,
        ncol=3,
        legend_title="Monitoring availability",
        file_title="fig5b_monitoring_risk.jpg",
        plots_path=plots_path,
        x_cat_order=risk_order,
        bar_groups=bar_groups,
        custom_cmap=custom_cmap,
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
        os.path.join(plots_path, "fig5b_monitoring_risk.csv"),
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

    # Remove row with index equal to "Global"
    multi_hazard_shm_results = multi_hazard_shm_results.drop("Global")
    multi_hazard_com_results = multi_hazard_com_results.drop("Global")

    # Function to adjust label positions to avoid overlap
    def adjust_text_positions(texts, offset=0.1):
        positions = [text.get_position() for text in texts]
        labels = [text.get_text() for text in texts]
        for i in range(len(positions) - 1):
            for j in range(i + 1, len(positions)):
                if (
                    labels[i] != labels[j]
                    and abs(positions[i][1] - positions[j][1]) < offset
                ):
                    positions[j] = (positions[j][0], positions[j][1] + offset)
                    texts[j].set_position(positions[j])

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

    # Plot the boxplot
    plt.figure(figsize=(7, 6))
    ax = sns.boxplot(
        x="Risk Value",
        y="Region",
        hue="Risk Type",
        data=long_df,
        width=0.5,
        # gap=0.2,
        boxprops=dict(alpha=0.7),
        showmeans=True,
        meanprops={
            "marker": "x",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": "4",
        },
    )

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.grid(True, which="both", color="grey", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=16)

    # Set y-ticks and y-tick labels
    y_ticks = np.arange(len(continent_labels))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(continent_labels)
    ax.tick_params(axis="y", labelsize=16)

    plt.xlabel("Risk Value", fontsize=18)
    plt.ylabel("Region (total number of bridges)", fontsize=18)

    # Change the text on the labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["SHM Risk", "Combined Monitoring Risk"]
    ax.legend(
        handles,
        new_labels,
        title="Risk Type",
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=2,
        fontsize=16,
        columnspacing=0.8,
    )

    # Save the plot
    plt.savefig(
        os.path.join(plots_path, "fig6a_box_plot_risk.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close(fig)

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
        os.path.join(plots_path, "fig6a_box_plot_risk.csv"),
        index=False,
    )

    # Plot relative change on a map
    world_geometries = gpd.read_file(
        "/mnt/g/RISK_PAPER/world_boundaries_for_plot/world-administrative-boundaries.shp"
    )

    # Define the mapping dictionary
    region_mapping = {
        "Eastern Asia": "Asia",
        "Central Asia": "Asia",
        "Southern Asia": "Asia",
        "South-Eastern Asia": "Asia",
        "Western Europe": "Europe",
        "Southern Europe": "Europe",
        "Northern Europe": "Europe",
        "Eastern Europe": "Europe",
        "Western Asia": "Middle East",
        "Northern America": "North America",
        "South America": "Latin America/Caribbean",
        "Central America": "Latin America/Caribbean",
        "Caribbean": "Latin America/Caribbean",
        "Northern Africa": "Africa",
        "Western Africa": "Africa",
        "Middle Africa": "Africa",
        "Eastern Africa": "Africa",
        "Southern Africa": "Africa",
        "Micronesia": "Oceania",
        "Polynesia": "Oceania",
        "Australia and New Zealand": "Oceania",
        "Melanesia": "Oceania",
    }

    # Add the region_lsb column
    world_geometries["region_lsb"] = world_geometries["region"].apply(
        lambda x: region_mapping.get(x, "Other")
    )

    # Remove everything that has region_lsb == Other
    world_geometries = world_geometries[world_geometries["region_lsb"] != "Other"]

    # Change Afghanistan, Iran, and Pakistan as they are also classsified as
    # Middle East in long-span brdgs db
    world_geometries.loc[world_geometries["name"] == "Afghanistan", "region_lsb"] = (
        "Middle East"
    )
    world_geometries.loc[
        world_geometries["name"] == "Iran (Islamic Republic of)", "region_lsb"
    ] = "Middle East"
    world_geometries.loc[world_geometries["name"] == "Pakistan", "region_lsb"] = (
        "Middle East"
    )

    # Create a separate gdf that contains only the regions
    world_geometries_regions = world_geometries.dissolve(by="region_lsb")

    world_geometries_regions = world_geometries_regions[["geometry"]]

    # compute centroids for annotations
    world_geometries_regions_projected = world_geometries_regions.to_crs(epsg=3035)
    world_geometries_regions_projected["centroid"] = (
        world_geometries_regions_projected.geometry.centroid
    )
    world_geometries_regions["centroid"] = world_geometries_regions_projected[
        "centroid"
    ].to_crs(world_geometries_regions.crs)

    # Assign percentage change to regions (disregard global for now)
    world_geometries_regions.loc[:, "percentage_change"] = risk_percentage_change[:-1]

    world_geometries_regions.rename(
        index={
            "Middle East": "Middle \nEast",
            "Latin America/Caribbean": "Latin America \nand Caribbean",
        },
        inplace=True,
    )

    regions_to_annotate = [
        "Africa",
        "Asia",
        "Europe",
        "Latin America \nand Caribbean",
        "Middle \nEast",
        "North America",
        "Oceania",
    ]

    adjustments = {
        "Africa": (3, -10),
        "Asia": (0, -5),
        "Europe": (-50, -28),
        "Latin America \nand Caribbean": (0, -15),
        "Middle \nEast": (0, -20),
        "North America": (-5, -20),
        "Oceania": (0, -13),
    }

    # initialize the figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 10))

    # create the plot
    world_geometries.plot(
        column="region_lsb", ax=ax, edgecolor="grey", linewidth=0.2, alpha=0.5
    )

    # custom axis
    # ax.set_xlim(-15, 35)
    # ax.set_ylim(32, 72)
    ax.axis("off")

    # annotate regions
    for region in regions_to_annotate:

        # get centroid
        centroid = world_geometries_regions.loc[
            world_geometries_regions.index == region, "centroid"
        ].values[0]
        x, y = centroid.coords[0]

        # get corrections
        x += adjustments[region][0]
        y += adjustments[region][1]

        print(region, x, y)

        # get rate and annotate
        rate = world_geometries_regions.loc[
            world_geometries_regions.index == region, "percentage_change"
        ].values[0]
        ax.annotate(
            f"{region}\n{rate:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=10,
            fontfamily="DejaVu Sans",
            color="black",
        )

    # Add global
    ax.annotate(
        f"Global {risk_percentage_change[-1]:.2f}%",
        (30, -60),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
        fontsize=10,
        fontfamily="DejaVu Sans",
        color="black",
    )

    # display the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "fig6b_map_relative_change.jpg"),
        dpi=600,
        bbox_inches="tight",
        # pad_inches=0,
    )

    plt.close()

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
        os.path.join(plots_path, "fig6b_map_relative_change.csv"),
        index=False,
    )

    # Plot map with bridge locations

    # initialize the figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 14))

    # create the plot
    world_geometries.plot(
        column="region_lsb",
        ax=ax,
        edgecolor="grey",
        linewidth=0.2,
        alpha=0.5,
        legend=True,
        legend_kwds={
            "title": "Region",
            "fontsize": 6,
            "title_fontsize": 7,
            "bbox_to_anchor": (0.07, -0.1),
            "loc": "lower left",
            "ncol": 7,
            "columnspacing": 0.8,
        },
    )
    bridges_gdf.plot(
        ax=ax,
        facecolor="black",
        edgecolor="black",
        linewidth=1,
        markersize=1,
        marker=".",
    )

    # custom axis
    # ax.set_xlim(-15, 35)
    # ax.set_ylim(32, 72)
    ax.axis("off")

    # display the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_path, "figSup2_map_bridges_locations.jpg"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.close()

    # Save raw data
    bridges_gdf[["ID", "Region", "geometry"]].to_csv(
        os.path.join(plots_path, "figSup2_map_bridges_locations.csv"),
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

    # Apply the color palette to the plot
    colors = [color_palette_SNT[category] for category in desired_order]

    plt.figure(figsize=(7, 5))
    ax = df_snt_availability_percentages.plot(
        kind="barh", stacked=True, ax=plt.gca(), color=colors
    )
    plt.xlabel("Share of the total number of bridges", fontsize=18)
    plt.xticks(
        plt.gca().get_xticks(),
        [f"{int(x * 100)}%" for x in plt.gca().get_xticks()],
        fontsize=16,
    )
    plt.yticks(fontsize=16)
    plt.xlim(0, 1)

    # Plot the legend below the plot
    # new_labels = ["No availability", "Low", "Medium", "High", "Very high"]
    plt.legend(
        title="Sentinel-1 availability",
        # labels=new_labels,
        bbox_to_anchor=(-0.55, -0.45),
        loc="lower left",
        ncol=5,
        fontsize=16,
        columnspacing=0.8,
    )

    # Remove the borders around the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add the share as a number in the middle of each stacked bar
    for container in ax.containers:
        labels = [f"{v * 100:.0f}%" if v != 0 else "" for v in container.datavalues]

        ax.bar_label(
            container,
            labels=labels,
            label_type="center",
            fontsize=14,
            color="black",
        )

    plt.savefig(
        os.path.join(plots_path, "fig7a_sentinel_availability.jpg"),
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

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
        os.path.join(plots_path, "fig7a_sentinel_availability.csv"),
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

    # Define color dictionary for SNT_availability_change
    color_dict_snt_change = {
        1: "#33BBEE",
        2: "#0077BB",
        3: "#EE7733",
        4: "#009988",
        5: "#CC3311",
    }

    # Define markers for each category
    markers = {
        1: "o",  # Circle
        2: "s",  # Square
        3: "^",  # Triangle
        4: "P",  # Plus
        5: "D",  # Diamond
    }

    # Define bounds and tick labels for SNT_availability_change
    bounds_snt_change = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    tick_labels_snt_change = [
        "Lost 1 satellite ({})".format(snt_change_counted["Lost 1 s/c"]),
        "Increased time ({})".format(snt_change_counted["Increased time"]),
        "No change ({})".format(snt_change_counted["No change"]),
        "Availability lost\ncompletely ({})".format(
            snt_change_counted["Availability lost completely"]
        ),
        "Covered by 2 satellites\ninstead of 1 ({})".format(
            snt_change_counted["Increased s/c availability"]
        ),
    ]
    tick_locations_snt_change = [1, 2, 3, 4, 5]

    # Plot map for SNT_availability_change
    col_title = "SNT_availability_change_num"
    plot_title = "SNT Availability Change"
    color_dict = color_dict_snt_change
    bounds = bounds_snt_change
    tick_labels = tick_labels_snt_change
    tick_locations = tick_locations_snt_change

    # Create a map
    fig, ax = plt.subplots(
        figsize=(7, 14), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_global()
    ax.coastlines(edgecolor="gray", alpha=0.2)
    ax.add_feature(cfeature.BORDERS, edgecolor="gray", alpha=0.1)

    # Define the colormap
    cmap = mcolors.ListedColormap(list(color_dict.values()))

    # Define the color boundaries
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot each category separately with different markers
    for category, marker in markers.items():
        subset = bridges_gdf[bridges_gdf[col_title] == category]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            label=f"{reverse_mapping[category]} ({snt_change_counted[reverse_mapping[category]]})",
            marker=marker,
            color=color_dict_snt_change[category],
            s=3,  # Adjust marker size as needed
            zorder=2,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}
    gl.top_labels = False
    gl.right_labels = False

    # Add a legend instead of a colorbar
    ax.legend(
        title="SNT Availability Change",
        title_fontsize=8,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,  # Adjust the number of columns in the legend
        columnspacing=0.8,
    )

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)

    ax.set_xlim([-150, 180])
    ax.set_ylim([-60, 75])

    # ax.set_title(plot_title, fontsize=20)

    plt.savefig(
        os.path.join(plots_path, f"fig7b_map_{col_title}.jpg"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

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
        os.path.join(plots_path, "fig7b_map_snt_availability_change.csv"),
        index=False,
    )
