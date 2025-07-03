"""
This module contains helper functions for plots for the risk calculation analysis.
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_map(
    bridges_gdf,
    col_title,
    plot_title,
    color_dict,
    bounds,
    tick_labels,
    tick_locations,
    plots_path,
):
    """
    Plot a map for the given column and save it as an image.

    Parameters:
    bridges_gdf (GeoDataFrame): The GeoDataFrame containing the data.
    col_title (str): The column title to plot.
    plot_title (str): The title of the plot.
    color_dict (dict): The color dictionary for the column.
    bounds (list): The color boundaries for the colormap.
    tick_labels (list): The tick labels for the colorbar.
    tick_locations (list): The tick locations for the colorbar.
    plots_path (str): The path to save the plot images.
    """
    # Create a map
    fig, ax = plt.subplots(
        figsize=(15, 20), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_global()
    ax.coastlines(edgecolor="gray", alpha=0.2)
    ax.add_feature(cfeature.BORDERS, edgecolor="gray", alpha=0.1)

    # Define the colormap
    cmap = mcolors.ListedColormap(list(color_dict.values()))

    # Define the color boundaries
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    scatter = ax.scatter(
        bridges_gdf.geometry.x,
        bridges_gdf.geometry.y,
        c=bridges_gdf[col_title],
        cmap=cmap,
        norm=norm,
        marker=".",
        s=5,
        zorder=2,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.2, boundaries=bounds, ticks=bounds)
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim([-150, 180])
    ax.set_ylim([-60, 75])

    ax.set_title(plot_title, fontsize=20)

    plt.savefig(
        os.path.join(plots_path, f"map_{col_title}.jpg"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()


# Function to plot pie chart with custom labels, colors, and order
def plot_pie_chart(
    data, title, filename, plots_path, label_mapping, color_mapping, sort_order=None
):
    # Sort the data if sort_order is provided
    if sort_order:
        data = data.reindex(sort_order)

    # Map the original values to the desired labels
    labels = [label_mapping.get(value, value) for value in data.index]
    # Map the labels to the desired colors
    colors = [color_mapping.get(label, "#ffffff") for label in labels]

    # Plot the pie chart with the new labels and colors
    fig, ax = plt.subplots(figsize=(12, 12))
    data.plot.pie(
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("")
    plt.title(title)
    plt.savefig(os.path.join(plots_path, filename))
    plt.close()


def plot_histogram(bridges_gdf, bin_edges, col, ylim, plots_path):
    # Plot histogram of the specified column
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate the histogram with weights to normalize the data
    weights = np.ones(len(bridges_gdf[f"{col}"])) / len(bridges_gdf[f"{col}"])
    _, _, patches = ax.hist(
        bridges_gdf[f"{col}"], bins=bin_edges, rwidth=0.8, weights=weights
    )

    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(f"{col}")
    ax.set_ylabel("Percentage of bridges")
    ax.set_ylim(0, ylim)

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # Calculate the actual counts without weights
    counts, _ = np.histogram(bridges_gdf[f"{col}"], bins=bin_edges)

    # Add counts on top of each bar
    for i in range(len(patches)):
        height = patches[i].get_height()
        count = counts[i]
        ax.annotate(
            f"{count}",
            xy=(patches[i].get_x() + patches[i].get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.savefig(
        os.path.join(plots_path, f"histogram_{col}.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure to free up memory


def plot_multiple_histograms(bridges_gdf, bin_edges, columns, ylim, plots_path, plot_title=None):
    """
    This function is used to plot multiple histograms of the specified columns
    as bars next to each other for given category.

    Arguments:
        bridges_gdf (GeoDataFrame): The GeoDataFrame containing the data.
        bin_edges (list): The bin edges for the histograms.
        columns (list): The list of columns to plot.
        ylim (int): The y-axis limit for the plot.
        plots_path (str): The path to save the plot images.
        plot_title (str): The title of the plot.

    Returns:
    None
    """

    title_str = " and ".join(columns)
    file_name_str = "_".join(columns)
    # Plot multiple histograms of the specified columns
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for each column
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    weights = np.ones(len(bridges_gdf[columns])) / len(bridges_gdf[columns])
    weights_proper_shape = np.tile(weights[:, np.newaxis], (1, len(columns)))
    _, _, patches = ax.hist(
        bridges_gdf[columns].to_numpy(),
        bins=bin_edges,
        rwidth=0.8,
        weights=weights_proper_shape,
        label=columns,
        color=colors,
        alpha=0.7,
    )

    ax.set_title(f"Histograms of {title_str}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Percentage of bridges")
    ax.set_ylim(0, ylim)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left")

    # Calculate the actual counts without weights for each column
    for col_idx, col in enumerate(columns):
        counts, _ = np.histogram(bridges_gdf[col], bins=bin_edges)
        # Add counts on top of each bar
        for i in range(len(patches[col_idx])):
            height = patches[col_idx][i].get_height()
            count = counts[i]
            ax.annotate(
                f"{count}",
                xy=(
                    patches[col_idx][i].get_x() + patches[col_idx][i].get_width() / 2,
                    height,
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Save the plot
    if plot_title is None:
        plt.savefig(
            os.path.join(plots_path, f"histogram_{file_name_str}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(plots_path, f"histogram_{plot_title}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)  # Close the figure to free up memory


def plot_bridge_parameter_histogram(
    df,
    ps_column,
    parameter_column,
    bins,
    labels,
    plot_path,
    plot_filename,
    bridge_parameters=None,
    fontsize=10,
    display_percentage=True,  # New parameter to define whether to display percentage or count
):
    """
    Plots a histogram showing the share of bridges by PS availability and a specified bridge parameter.

    Parameters:
    - df: DataFrame containing the data.
    - ps_column: Column name for PS availability.
    - parameter_column: Column name for the bridge parameter.
    - bins: List of bin edges for PS availability.
    - labels: List of labels for the bins.
    - plot_path: Path to save the plot.
    - plot_filename: Filename for the saved plot.
    - bridge_parameters: List of bridge parameters to specify the order of plotting.
        If None, use unique values from the parameter_column.
    - fontsize: Font size for the numbers on the bars.
    - display_percentage: Boolean to define whether to display percentage or count at the top of each bar.
    """

    # Define the unique bridge parameters if not provided
    if bridge_parameters is None:
        # Replace None values in the df[parameter_column] with 'Unknown'
        df[parameter_column] = df[parameter_column].fillna("Unknown")
        bridge_parameters = df[parameter_column].unique()

    # Categorize the PS availability data into the defined bins
    df["PS_availability_bin"] = pd.cut(
        df[ps_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    # Group the data by the bins and bridge parameters
    grouped_data = (
        df.groupby(["PS_availability_bin", parameter_column])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize the counts to get the share of bridges if display_percentage is True
    if display_percentage:
        grouped_data = grouped_data.div(grouped_data.sum(axis=0), axis=1)

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the size as needed

    # Define bar width and positions
    bar_width = 0.15
    x = np.arange(len(labels))

    # Plot bars for each bridge parameter
    for i, parameter in enumerate(bridge_parameters):
        bars = ax.bar(
            x + i * bar_width, grouped_data[parameter], bar_width, label=parameter
        )
        # Add numbers at the top of each bar
        for bar in bars:
            height = bar.get_height()
            if display_percentage:
                text = f"{height:.0%}"
            else:
                text = f"{int(height)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                text,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    # Add labels and title
    ax.set_xlabel("PS Availability")
    ax.set_ylabel("Share of Bridges" if display_percentage else "Count of Bridges")
    ax.set_title(f"Share of Bridges by PS Availability and {parameter_column}")
    ax.set_xticks(x + bar_width * (len(bridge_parameters) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    # Save the plot
    plt.savefig(
        os.path.join(plot_path, plot_filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure to free up memory


def calculate_counts(df, col_title, bins, bin_labels, continents):
    """
    Calculate the counts of bridges in each bin for each continent.
    """
    df_counts = pd.DataFrame(index=continents, columns=bin_labels).fillna(0)

    for continent in continents:
        continent_data = df[df["Region"] == continent]
        binned_data = pd.cut(
            continent_data[col_title],
            bins=bins,
            labels=bin_labels,
            include_lowest=True,
        )
        bin_counts = binned_data.value_counts()
        for bin_label in bin_labels:
            df_counts.loc[continent, bin_label] = bin_counts.get(bin_label, 0)

    return df_counts


def calculate_percentages(df, col_title, bins, bin_labels, continents):
    """
    Calculate the percentages of bridges in each bin for each continent.
    """
    df_percentages = (
        pd.DataFrame(index=continents, columns=bin_labels).astype(float).fillna(0.0)
    )

    for continent in continents:
        continent_data = df[df["Region"] == continent]
        binned_data = pd.cut(
            continent_data[col_title],
            bins=bins,
            labels=bin_labels,
            include_lowest=True,
        )
        bin_counts = binned_data.value_counts()
        total_bridges = len(continent_data)
        for bin_label in bin_labels:
            count = bin_counts.get(bin_label, 0)
            percentage = (count / total_bridges) * 100 if total_bridges > 0 else 0
            df_percentages.loc[continent, bin_label] = percentage

    return df_percentages


def save_to_csv(df_list, file_path):
    """
    Save the combined DataFrame to a CSV file.
    """
    combined_df = pd.concat(df_list)
    combined_df.to_csv(file_path)
    print(f"All counts saved to {file_path}")


def save_to_excel(monitoring_results, file_path):
    """
    Save the DataFrames to an Excel file with separate sheets.
    """
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for col_title, df_percentages in monitoring_results.items():
            df_percentages.to_excel(writer, sheet_name=col_title)
    print(f"All percentages saved to {file_path}")


def create_class_column(
    df,
    source_col,
    target_col,
    bins,
    labels,
    right=False,
    include_lowest=True,
    duplicates="raise",
):
    """
    Create a new column that assigns classes based on a continuous value.

    Parameters:
    - df: DataFrame containing the data.
    - source_col: Column name for the source continuous values.
    - target_col: Column name for the target class values.
    - bins: List of bin edges.
    - labels: List of labels for the bins.
    - right: Indicates whether bins include the rightmost edge or not.
    - include_lowest: Whether the first interval should be left-inclusive or not.
    - duplicates: How to handle bin edges that are not unique.
    """
    df[target_col] = pd.cut(
        df[source_col],
        bins=bins,
        labels=labels,
        right=right,
        include_lowest=include_lowest,
        duplicates=duplicates,
    )


def calculate_monitoring_results(df, col_list, index_col):
    """
    Calculate the number of bridges in each monitoring class for each region.
    """
    results = pd.DataFrame(index=df[index_col].unique())
    for col in col_list:

        for region in results.index:
            region_data = df[df[index_col] == region]
            region_counts = region_data[col].value_counts()
            for monitoring_class in region_counts.index:
                results.loc[region, monitoring_class] = region_counts.get(
                    monitoring_class, 0
                )
    results = results.fillna(0).astype(int)
    results.loc["Global"] = results.sum()
    return results


def save_table_as_image(df, title, file_path, figsize=(10, 6), fontsize=10):
    """
    Save a DataFrame as a table image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    ax.set_title(title, fontsize=16)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_vulnerability_monitoring_results(
    df, vulnerability_order, vulnerability_col, monitoring_cols
):
    """
    Calculate the number of bridges in each monitoring class for each vulnerability level.
    """
    results = pd.DataFrame(index=vulnerability_order)
    for vulnerability in vulnerability_order:
        vulnerability_data = df[df[vulnerability_col] == vulnerability]
        for col in monitoring_cols:
            counts = vulnerability_data[col].value_counts()
            for monitoring_class in counts.index:
                results.loc[vulnerability, f"{monitoring_class}"] = counts.get(
                    monitoring_class, 0
                )
    results = results.fillna(0).astype(int)
    return results


def plot_stacked_histogram(
    df,
    x_labels,
    bar_groups,
    bar_width=0.3,
    bar_spacing=0.1,
    xlabel="X-axis",
    ylabel="Y-axis",
    title="Stacked Histogram",
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.05, 1),
    save_path=None,
    figsize=(12, 8),
    dpi=300,
    alpha=0.7,
):
    """
    Plot a stacked histogram.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_labels: List of labels for the x-axis.
    - bar_groups: List of tuples, where each tuple contains the column names and colors for a group of bars.
    - bar_width: Width of each bar.
    - bar_spacing: Spacing between bar groups.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - legend_loc: Location of the legend.
    - legend_bbox_to_anchor: Bounding box anchor for the legend.
    - save_path: Path to save the plot image. If None, the plot is not saved.
    - figsize: Size of the figure.
    - dpi: Dots per inch for the saved image.
    - alpha: Transparency level for the bars.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(x_labels))

    for i, (cols, colors) in enumerate(bar_groups):
        bottom = np.zeros(len(x_labels))
        for col, color in zip(cols, colors):
            ax.bar(
                x
                + (i * (bar_width + bar_spacing))
                - ((len(bar_groups) - 1) * (bar_width + bar_spacing)) / 2,
                df[col],
                bar_width,
                bottom=bottom,
                label=col,
                color=color,
                alpha=alpha,
            )
            bottom += df[col]

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_histogram_new(
    df,
    x_labels,
    bar_groups,
    bar_width=0.3,
    bar_spacing=0.1,
    xlabel="X-axis",
    ylabel="Y-axis",
    title="Stacked Histogram",
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.05, 1),
    save_path=None,
    figsize=(12, 8),
    dpi=300,
    alpha=0.7,
):
    """
    Plot a stacked histogram.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_labels: List of labels for the x-axis.
    - bar_groups: List of tuples, where each tuple contains the column names and colors for a group of bars.
    - bar_width: Width of each bar.
    - bar_spacing: Spacing between bar groups.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - legend_loc: Location of the legend.
    - legend_bbox_to_anchor: Bounding box anchor for the legend.
    - save_path: Path to save the plot image. If None, the plot is not saved.
    - figsize: Size of the figure.
    - dpi: Dots per inch for the saved image.
    - alpha: Transparency level for the bars.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(x_labels))

    for i, (cols, colors) in enumerate(bar_groups):
        bottom = np.zeros(len(x_labels))
        for col, color in zip(cols, colors):
            bars = ax.bar(
                x
                + (i * (bar_width + bar_spacing))
                - ((len(bar_groups) - 1) * (bar_width + bar_spacing)) / 2,
                df[col],
                bar_width,
                bottom=bottom,
                label=col,
                color=color,
                alpha=alpha,
            )
            bottom += df[col]
            bar_heights = [bar.get_height() for bar in bars]

            if col == cols[0]:
                bar_heights_cumulative = bar_heights
            else:
                bar_heights_cumulative = np.array(bar_heights_cumulative) + np.array(
                    bar_heights
                )

            # If last group, add labels on top of the bars
            if col == cols[-1]:
                # Add cumulative labels to the bars
                ax.bar_label(
                    bars,
                    labels=[f"{int(height)}" for height in bar_heights_cumulative],
                    padding=3,
                )

                # ax.bar_label(bars, labels=bar_heights_cumulative, padding=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
