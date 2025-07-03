import os

import numpy as np
import pandas as pd

# import geopandas as gpd

import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba

import ptitprince as pt
import seaborn as sns


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


def set_fonts_for_figure(figsize):
    # Scale fonts based on figure area compared to 10x6 reference
    scale = np.sqrt((figsize[0] * figsize[1]) / (10 * 6))
    base_font = 16 * scale

    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.titlesize": base_font * 1.125,
            "axes.labelsize": base_font * 1.125,
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "legend.fontsize": base_font,
            "figure.titlesize": base_font * 1.25,
        }
    )


def plot_ps_availability_with_distribution(
    bridges_gdf, column_name, desired_order, custom_cmap, plots_path, figure_id
):
    # Extract colors from the colormap
    colors_ordered = custom_cmap.colors

    # Define the availability labels in the same order as your colormap
    availability_labels = [
        "[0, 0.2]",
        "(0.2, 0.4]",
        "(0.4, 0.6]",
        "(0.6, 0.8]",
        "(0.8, 1.0]",
    ]

    one_colour = [to_rgba("#4A7A95", 0.8)]  # Use a single color for the raincloud plot

    # Set font sizes for the figure
    set_fonts_for_figure((12, 12))

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Create raincloud plot for all materials
    pt.RainCloud(
        x=column_name,
        y="Monitoring_PS_availability_only",
        data=bridges_gdf,
        palette=one_colour,
        bw=0.15,
        width_viol=0.4,
        ax=ax,
        orient="h",
        move=0,
    )

    # Now overlay histogram bars for each material
    bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    bar_width = 0.15
    bar_height_scale = 0.6  # Maximum height scaling for histogram bars

    for i, material in enumerate(desired_order):
        material_data = bridges_gdf[bridges_gdf[column_name] == material]

        if len(material_data) > 0:
            # Calculate proportions for this material
            proportions = material_data[
                "Monitoring_PS_availability_classes"
            ].value_counts(normalize=True)
            proportions = proportions.reindex(availability_labels, fill_value=0)

            # Y position - bottom level where all bars start
            y_bottom = i - 0.5  # Bottom edge of all bars for this material

            # Create histogram bars above the raincloud for this material
            for j, (label, prop) in enumerate(proportions.items()):
                if prop > 0:
                    # Calculate actual bar height
                    bar_actual_height = bar_height_scale * prop

                    # Position bar so its bottom starts at y_bottom and extends upward
                    bar_y_position = y_bottom - bar_actual_height / 2

                    # Create bar
                    _ = ax.barh(
                        y=bar_y_position,
                        width=bar_width,
                        height=-bar_actual_height,
                        left=bin_centers[j] - bar_width / 2,
                        color=colors_ordered[j],
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=1,
                    )

                    # Add percentage label at center of bar
                    ax.text(
                        bin_centers[j],
                        y_bottom - 0.1,
                        f"{prop:.0%}",
                        ha="center",
                        va="center",
                    )

    # Customize the plot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(len(desired_order) - 0.75, -0.85)  # Extended to accommodate bars above
    ax.set_xlabel("PS availability")
    ax.set_ylabel(f"{column_name}")

    # Update y-tick labels to include bridge counts
    y_labels = []
    for material in desired_order:
        count = len(bridges_gdf[bridges_gdf[column_name] == material])
        y_labels.append(f"{material}\n({count} bridges)")

    ax.set_yticks([i - 0.3 for i in range(len(desired_order))])
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis="x", alpha=0.3)

    # Add horizontal lines for bar plots
    for i in range(-1, len(desired_order)):
        ax.hlines(
            y=i + 0.5,
            xmin=0,
            xmax=1,
            color="grey",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )

    # Add horizontal lines between materials
    for i in range(-2, len(desired_order)):
        ax.axhline(y=i + 1.2, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Add legend below the plot in 3 columns
    nice_labels = ["Very low", "Low", "Medium", "High", "Very high"]

    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, color=colors_ordered[i], alpha=0.8, label=nice_labels[i]
        )
        for i in range(len(nice_labels))
    ]

    ax.legend(
        handles=legend_elements,
        title="PS availability classes",
        bbox_to_anchor=(0.4, -0.08),
        loc="upper center",
        ncol=len(nice_labels),  # All 5 items in one row
        handlelength=1.2,
        columnspacing=0.4,
        handletextpad=0.2,
    )

    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plots_path,
            f"{figure_id}_ps_availability_distribution_by_{column_name.lower()}.jpg",
        ),
        bbox_inches="tight",
        dpi=600,
    )
    plt.show()
    plt.close()
