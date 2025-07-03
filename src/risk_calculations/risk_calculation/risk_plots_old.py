"""
This script can be used to create plots from the data in the 'Bridges' GeoDataFrame.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature

if __name__ == "__main__":

    # Define the path to the data
    data_path = "/mnt/g/RISK_PAPER"

    # Read the bridges gdf from a pickle
    bridges_gdf = pd.read_pickle(
        os.path.join(data_path, "lsb_risk_analysis_filtered.pkl")
    )

    # Set the font size
    plt.rcParams["font.size"] = 18

    # # PS DENSITY ANALYSIS #
    # # Create a figure with 8 plots in two rows and four columns
    # # Each subplot represents a continent + the last one is a cummulative one over the whole world
    # # Plot histogram of the ps_mean_{}_1 values for each continent
    # # bin_edges = [0, 1, 5, 10, 15, 5000000]
    # bin_edges = [0, 1, 5, 10, 20, 150]
    # positions = np.arange(len(bin_edges) - 1)

    # # Plot histogram of the ps_mean_world_{} values for each segment
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Define the width of the bars and the positions of the bars for each group
    # width = 0.15

    # for i in range(1, 6):
    #     data = bridges_gdf["ps_mean_world_{}".format(i)]
    #     data = data.dropna()
    #     frequencies, _ = np.histogram(data, bins=bin_edges)

    #     # Convert frequencies to percentages
    #     percentages = (frequencies / len(data)) * 100

    #     # Plot the bars for this group
    #     ax.bar(
    #         positions - 2 * width + i * width,
    #         percentages,
    #         width=width,
    #         alpha=0.5,
    #         label="Segment {}".format(i),
    #     )

    # ax.set_ylim([0, 100])  # Set the y-axis limits to [0, 70]
    # ax.set_ylabel("Percentage")  # Set the label of the y-axis
    # ax.set_title("World")  # Set the title of the plot
    # ax.legend()  # Add a legend

    # # Define the labels for the x-axis
    # labels = [
    #     "[{},{})".format(bin_edges[k - 1], bin_edges[k])
    #     for k in range(1, len(bin_edges))
    # ]
    # ax.set_xticks(positions)
    # ax.set_xticklabels(labels, rotation=45)

    # plt.suptitle("Distribution of PS mean values for each segment")
    # plt.tight_layout(
    #     rect=[0, 0, 1, 0.96]
    # )  # Adjust the layout to make room for the suptitle
    # plt.savefig(
    #     os.path.join(data_path, "Plots", "hist_PS_mean_world_allsegments.jpg"),
    #     dpi=600,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # # Plot histogram of the ps_avail_perc_{} values for each segment
    # bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # positions = np.arange(len(bin_edges) - 1)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Define the width of the bars and the positions of the bars for each group
    # width = 0.15

    # for i in range(1, 6):
    #     data = bridges_gdf["ps_avail_perc_{}".format(i)]
    #     data = data.dropna()
    #     frequencies, _ = np.histogram(data, bins=bin_edges)

    #     # Convert frequencies to percentages
    #     percentages = (frequencies / len(data)) * 100

    #     # Plot the bars for this group
    #     ax.bar(
    #         positions - 2 * width + i * width,
    #         percentages,
    #         width=width,
    #         alpha=0.5,
    #         label="Segment {}".format(i),
    #     )

    # ax.set_ylim([0, 100])  # Set the y-axis limits to [0, 70]
    # ax.set_ylabel("Percentage")  # Set the label of the y-axis
    # ax.set_title("World")  # Set the title of the plot
    # ax.legend()  # Add a legend

    # # Define the labels for the x-axis
    # labels = [
    #     "[{},{})".format(bin_edges[k - 1], bin_edges[k])
    #     for k in range(1, len(bin_edges))
    # ]
    # ax.set_xticks(positions)
    # ax.set_xticklabels(labels, rotation=45)

    # plt.suptitle("Distribution of PS density values for each segment")
    # plt.tight_layout(
    #     rect=[0, 0, 1, 0.96]
    # )  # Adjust the layout to make room for the suptitle
    # plt.savefig(
    #     os.path.join(data_path, "Plots", "hist_PS_avail_world_allsegments.jpg"),
    #     dpi=600,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # bridge_types = ["Suspension", "Arch", "Cantilever", "Cable Stayed", "Truss"]

    # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

    # bridge_types.append("World")  # Add 'World' to the list of bridge types

    # for i, bridge_type in enumerate(bridge_types):
    #     ax = axs[i // 3, i % 3]  # Determine the subplot to use

    #     if bridge_type == "World":
    #         data_df = bridges_gdf  # Use all data for 'World'
    #     else:
    #         data_df = bridges_gdf[
    #             bridges_gdf["Type"] == bridge_type
    #         ]  # Filter the data by bridge type

    #     for j in range(1, 6):
    #         data = data_df["ps_mean_world_{}".format(j)]
    #         data = data.dropna()
    #         frequencies, _ = np.histogram(data, bins=bin_edges)

    #         # Convert frequencies to percentages
    #         percentages = (frequencies / len(data)) * 100

    #         # Plot the bars for this group
    #         ax.bar(
    #             positions - 2 * width + j * width,
    #             percentages,
    #             width=width,
    #             alpha=0.5,
    #             label="Segment {}".format(j),
    #         )

    #     ax.set_ylim([0, 60])  # Set the y-axis limits to [0, 50]
    #     ax.set_ylabel("Percentage")  # Set the label of the y-axis
    #     ax.set_title(bridge_type)  # Set the title of the subplot
    #     # ax.legend()  # Add a legend

    #     # Define the labels for the x-axis
    #     labels = [
    #         "[{},{})".format(bin_edges[k - 1], bin_edges[k])
    #         for k in range(1, len(bin_edges))
    #     ]
    #     ax.set_xticks(positions)
    #     ax.set_xticklabels(labels, rotation=45)

    # plt.suptitle("Distribution of PS density values for each segment")

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=5)

    # plt.tight_layout(
    #     rect=[0, 0.03, 1, 0.96]
    # )  # Adjust the layout to make room for the suptitle and legend

    # plt.savefig(
    #     os.path.join(data_path, "Plots", "hist_PS_mean_bridge_types_allsegments.jpg"),
    #     dpi=600,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # bridge_lengths = [500, 1000, 1500, 2000, 3000, 1000000]

    # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

    # for i, bridge_length in enumerate(bridge_lengths):
    #     ax = axs[i // 3, i % 3]  # Determine the subplot to use

    #     if i == 0:
    #         data_df = bridges_gdf[bridges_gdf["Total Leng"] < bridge_length]
    #         plot_title = "0-{}".format(bridge_length)
    #     else:
    #         data_df = bridges_gdf[
    #             (bridges_gdf["Total Leng"] > bridge_lengths[i - 1])
    #             & (bridges_gdf["Total Leng"] < bridge_length)
    #         ]
    #         plot_title = "{}-{}".format(bridge_lengths[i - 1], bridge_length)

    #     for j in range(1, 6):
    #         data = data_df["ps_mean_world_{}".format(j)]
    #         data = data.dropna()
    #         frequencies, _ = np.histogram(data, bins=bin_edges)

    #         # Convert frequencies to percentages
    #         percentages = (frequencies / len(data)) * 100

    #         # Plot the bars for this group
    #         ax.bar(
    #             positions - 2 * width + j * width,
    #             percentages,
    #             width=width,
    #             alpha=0.5,
    #             label="Segment {}".format(j),
    #         )

    #     ax.set_ylim([0, 60])  # Set the y-axis limits to [0, 50]
    #     ax.set_ylabel("Percentage")  # Set the label of the y-axis
    #     ax.set_title(plot_title)  # Set the title of the subplot
    #     # ax.legend()  # Add a legend

    #     # Define the labels for the x-axis
    #     labels = [
    #         "[{},{})".format(bin_edges[k - 1], bin_edges[k])
    #         for k in range(1, len(bin_edges))
    #     ]
    #     ax.set_xticks(positions)
    #     ax.set_xticklabels(labels, rotation=45)

    # plt.suptitle("Distribution of PS density values for each segment")

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=5)

    # plt.tight_layout(
    #     rect=[0, 0.03, 1, 0.96]
    # )  # Adjust the layout to make room for the suptitle and legend

    # plt.savefig(
    #     os.path.join(data_path, "Plots", "hist_PS_mean_bridge_length_allsegments.jpg"),
    #     dpi=600,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # # Plot histogram with custom bin edges of the ps_sum column
    # # Define the bin edges
    # # bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 500, 1000, 5000000]
    # bin_edges = [0, 10, 50, 100, 200, 500, 5000000]

    # Plot on maps
    col_title = [
        "Exposure_norm",
        "Vulnerability_norm",
        "ls_max",
        "su_max",
        "Landslide_risk",
        "Subsidence_risk",
        "Landslide_risk_mon",
        "Subsidence_risk_mon",
        "Monitoring_class",
    ]
    plot_title = [
        "Exposure",
        "Vulnerability",
        "Landslide_hazard",
        "Subsidence_hazard",
        "Landslide_risk",
        "Subsidence_risk",
        "Landslide_risk_mon",
        "Subsidence_risk_mon",
        "Monitoring_class",
    ]

    # col_title = ['Exposure_norm', 'Vulnerability_norm']
    # plot_title = ['Exposure', 'Vulnerability']

    for i in range(len(col_title)):

        # Create a map
        fig, ax = plt.subplots(
            figsize=(15, 20), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        ax.set_global()
        ax.coastlines(edgecolor="gray", alpha=0.2)

        # ax.add_feature(cfeature.COASTLINE, edgecolor='gray', alpha=0.3)
        ax.add_feature(cfeature.BORDERS, edgecolor="gray", alpha=0.1)

        # color_dict = {0: '#FFFFFF', 0.2: '#A9D18E', 0.4: '#C5E0B4', 0.6: '#FFE699',
        # 0.8: '#FFC000', 1:'#FF8A8A'}  # Replace with your values and colors
        color_dict = {
            0: "#4575b4",
            0.2: "#91bfdb",
            0.4: "#e0f3f8",
            0.6: "#fee090",
            0.8: "#fc8d59",
            1: "#d73027",
        }  # Replace with your values and colors

        colors = bridges_gdf[col_title[i]].map(color_dict).tolist()

        # Define the colormap
        if col_title[i] == "Monitoring_class":
            color_dict_monit = {
                0: "#d73027",
                0.2: "#fc8d59",
                0.4: "#fee090",
                0.6: "#e0f3f8",
                0.8: "#91bfdb",
                1: "#4575b4",
            }  # Replace with your values and colors

            cmap = mcolors.ListedColormap(list(color_dict_monit.values()))
        else:
            cmap = mcolors.ListedColormap(list(color_dict.values()))

        # Define the color boundaries
        # bounds = list(color_dict.keys()) + [1.0]  # Add an extra boundary
        bounds = [-0.2, 0.001, 0.201, 0.401, 0.601, 0.801, 1.0]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        scatter = ax.scatter(
            bridges_gdf.geometry.x,
            bridges_gdf.geometry.y,
            c=bridges_gdf[
                col_title[i]
            ],  # Replace 'displacement_column' with the actual column name
            # cmap="jet",  # You can choose another colormap
            cmap=cmap,
            norm=norm,
            marker=".",
            #     edgecolors='k',  # Black edges for better visibility
            s=5,  # Marker size
            # vmin=0,
            # vmax=1,
            zorder=2,
        )

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.top_labels = False
        gl.right_labels = False

        # cbar = plt.colorbar(scatter, ax=ax, shrink=0.4)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, boundaries=bounds, ticks=bounds)
        cbar.set_ticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])  # Set the tick locations
        if col_title[i] == "Monitoring_class":
            cbar.set_ticklabels(
                ["No monitoring", "Low", "Medium-Low", "Medium", "Medium-High", "High"]
            )
        else:
            cbar.set_ticklabels(
                ["No risk", "Low", "Medium-Low", "Medium", "Medium-High", "High"]
            )
        cbar.set_label(plot_title[i], fontsize=16)
        cbar.ax.tick_params(labelsize=15)

        # Set axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # ### TESTING ON EUROPE ###
        # # Set the x and y limits
        # ax.set_xlim([-15, 50])
        # ax.set_ylim([34, 75])

        ax.set_title(plot_title[i], fontsize=20)

        # plt.savefig(
        #     data_path + r"/Plots" + r"/eu_map_{}_plot.jpg".format(plot_title[i]),
        #     dpi=600,
        #     bbox_inches="tight",
        # )

        plt.savefig(
            data_path + r"/Plots" + r"/map_{}_plot.jpg".format(plot_title[i]),
            dpi=600,
            bbox_inches="tight",
        )

        plt.close()

    # Define the bins
    bins = [-0.2, 0.00001, 0.2, 0.4, 0.6, 0.8, 1]

    # Create arrays for the 'Landslide_risk' and 'Subsidence_risk' values where 'monitoring' is 0 and 1
    landslide_risk_monitoring_0 = bridges_gdf[bridges_gdf["Monitoring"] == 0][
        "Landslide_risk"
    ]
    landslide_risk_monitoring_1 = bridges_gdf[bridges_gdf["Monitoring"] == 1][
        "Landslide_risk"
    ]

    subsidence_risk_monitoring_0 = bridges_gdf[bridges_gdf["Monitoring"] == 0][
        "Subsidence_risk"
    ]
    subsidence_risk_monitoring_1 = bridges_gdf[bridges_gdf["Monitoring"] == 1][
        "Subsidence_risk"
    ]

    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot stacked histogram of the landslide risk
    axs[0].hist(
        [landslide_risk_monitoring_0, landslide_risk_monitoring_1],
        bins=bins,
        stacked=True,
        color=["blue", "lightblue"],
        label=["No monitoring", "SHM"],
        rwidth=0.8,
        histtype="barstacked",
    )
    axs[0].set_title("Landslide risk")
    axs[0].set_xlabel("Risk")
    axs[0].set_ylabel("Count")
    axs[0].set_ylim([0, 400])
    axs[0].set_xlim([-0.3, 1.1])
    axs[0].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[0].set_xticklabels(
        ["No risk", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )
    axs[0].legend()

    # Plot stacked histogram of the subsidence risk
    axs[1].hist(
        [subsidence_risk_monitoring_0, subsidence_risk_monitoring_1],
        bins=bins,
        stacked=True,
        color=["red", "pink"],
        label=["No monitoring", "SHM"],
        rwidth=0.8,
        histtype="barstacked",
    )
    axs[1].set_title("Subsidence risk")
    axs[1].set_xlabel("Risk")
    axs[1].set_ylabel("Count")
    axs[1].set_ylim([0, 400])
    axs[1].set_xlim([-0.3, 1.1])
    axs[1].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[1].set_xticklabels(
        ["No risk", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )
    axs[1].legend()

    # Save the figure
    plt.savefig(
        data_path + r"/Plots" + r"/risk_histogram_SHM.jpg",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    bins_monit = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # Create bins of 'Monitoring_class'
    bridges_gdf["Monitoring_class_bins"] = pd.cut(
        bridges_gdf["Monitoring_class"], bins_monit
    )

    # Group data by 'Monitoring_class_bins'
    landslide_risk_monitoring = bridges_gdf.groupby("Monitoring_class_bins")[
        "Landslide_risk"
    ]
    subsidence_risk_monitoring = bridges_gdf.groupby("Monitoring_class_bins")[
        "Subsidence_risk"
    ]

    # Create a list of arrays for each bin
    landslide_risk_bins = [
        bridges_gdf.loc[bridges_gdf["Monitoring_class_bins"] == bin, "Landslide_risk"]
        for bin in bridges_gdf["Monitoring_class_bins"].cat.categories
    ]
    subsidence_risk_bins = [
        bridges_gdf.loc[bridges_gdf["Monitoring_class_bins"] == bin, "Subsidence_risk"]
        for bin in bridges_gdf["Monitoring_class_bins"].cat.categories
    ]

    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot stacked histogram of the landslide risk
    axs[0].hist(
        landslide_risk_bins,
        bins=bins,
        stacked=True,
        color=["#FF111B", "#FF8F07", "#FFD21C", "#A2DF44", "#97D6A7"],
        label=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"],
        rwidth=0.8,
        histtype="barstacked",
    )
    axs[0].set_title("Landslide risk")
    axs[0].set_xlabel("Risk")
    axs[0].set_ylabel("Count")
    axs[0].set_ylim([0, 400])
    axs[0].set_xlim([-0.3, 1.1])
    axs[0].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[0].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[0].legend(title="Monitoring")

    # Plot stacked histogram of the subsidence risk
    axs[1].hist(
        subsidence_risk_bins,
        bins=bins,
        stacked=True,
        color=["#FF111B", "#FF8F07", "#FFD21C", "#A2DF44", "#97D6A7"],
        label=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"],
        rwidth=0.8,
        histtype="barstacked",
    )
    axs[1].set_title("Subsidence risk")
    axs[1].set_xlabel("Risk")
    axs[1].set_ylabel("Count")
    axs[1].set_ylim([0, 400])
    axs[1].set_xlim([-0.3, 1.1])
    axs[1].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[1].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[1].legend(title="Monitoring")

    # Save the figure
    plt.savefig(
        data_path + r"/Plots" + r"/risk_histogram_SARmonitoring.jpg",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    # Create bins of 'Monitoring_class'
    bridges_gdf["Monitoring_class_bins"] = pd.cut(
        bridges_gdf["Monitoring_class"], bins_monit
    )

    # Group data by 'Monitoring_class_bins'
    landslide_risk_monitoring = bridges_gdf.groupby("Monitoring_class_bins")[
        "Landslide_risk_mon"
    ]
    subsidence_risk_monitoring = bridges_gdf.groupby("Monitoring_class_bins")[
        "Subsidence_risk_mon"
    ]

    # Create a list of arrays for each bin
    landslide_risk_bins = [
        bridges_gdf.loc[
            bridges_gdf["Monitoring_class_bins"] == bin, "Landslide_risk_mon"
        ]
        for bin in bridges_gdf["Monitoring_class_bins"].cat.categories
    ]
    subsidence_risk_bins = [
        bridges_gdf.loc[
            bridges_gdf["Monitoring_class_bins"] == bin, "Subsidence_risk_mon"
        ]
        for bin in bridges_gdf["Monitoring_class_bins"].cat.categories
    ]

    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot stacked histogram of the landslide risk
    axs[0].hist(
        landslide_risk_bins,
        bins=bins,
        stacked=True,
        color=["#FF111B", "#FF8F07", "#FFD21C", "#A2DF44", "#97D6A7"],
        label=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"],
        rwidth=0.8,
    )
    axs[0].set_title("Landslide risk")
    axs[0].set_xlabel("Risk")
    axs[0].set_ylabel("Count")
    axs[0].set_ylim([0, 400])
    axs[0].set_xlim([-0.3, 1.1])
    axs[0].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[0].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[0].legend(title="Monitoring")

    # Plot stacked histogram of the subsidence risk
    axs[1].hist(
        subsidence_risk_bins,
        bins=bins,
        stacked=True,
        color=["#FF111B", "#FF8F07", "#FFD21C", "#A2DF44", "#97D6A7"],
        label=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"],
        rwidth=0.8,
    )
    axs[1].set_title("Subsidence risk")
    axs[1].set_xlabel("Risk")
    axs[1].set_ylabel("Count")
    axs[1].set_ylim([0, 400])
    axs[1].set_xlim([-0.3, 1.1])
    axs[1].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[1].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[1].legend(title="Monitoring")

    # Save the figure
    plt.savefig(
        data_path + r"/Plots" + r"/risk_histogram_SARmonitoring_scaled.jpg",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    # Create arrays for the 'Landslide_risk' and 'Subsidence_risk' values where 'monitoring' is 0 and 1
    landslide_risk_monitoring_0 = bridges_gdf[bridges_gdf["snt_avail"] == 0][
        "Landslide_risk"
    ]
    landslide_risk_monitoring_5 = bridges_gdf[bridges_gdf["snt_avail"] == 0.5][
        "Landslide_risk"
    ]
    landslide_risk_monitoring_10 = bridges_gdf[bridges_gdf["snt_avail"] == 1][
        "Landslide_risk"
    ]

    subsidence_risk_monitoring_0 = bridges_gdf[bridges_gdf["snt_avail"] == 0][
        "Subsidence_risk"
    ]
    subsidence_risk_monitoring_5 = bridges_gdf[bridges_gdf["snt_avail"] == 0.5][
        "Subsidence_risk"
    ]
    subsidence_risk_monitoring_10 = bridges_gdf[bridges_gdf["snt_avail"] == 1][
        "Subsidence_risk"
    ]

    # Create two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot stacked histogram of the landslide risk
    axs[0].hist(
        [
            landslide_risk_monitoring_0,
            landslide_risk_monitoring_5,
            landslide_risk_monitoring_10,
        ],
        bins=bins,
        stacked=True,
        # color=['#97D6A7', '#A2DF44', '#FFD21C', '#FF8F07','#FF111B'],
        # color=[ '#97D6A7', '#A2DF44', '#FFD21C', '#FF8F07','#FF111B', '#8B0000'],
        label=["None", "Only one", "Both"],
        rwidth=0.8,
    )
    axs[0].set_title("Landslide risk")
    axs[0].set_xlabel("Risk")
    axs[0].set_ylabel("Count")
    axs[0].set_ylim([0, 400])
    axs[0].set_xlim([-0.3, 1.1])
    axs[0].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[0].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[0].legend()

    # Plot stacked histogram of the subsidence risk
    axs[1].hist(
        [
            subsidence_risk_monitoring_0,
            subsidence_risk_monitoring_5,
            subsidence_risk_monitoring_10,
        ],
        bins=bins,
        stacked=True,
        # color=['#97D6A7', '#A2DF44', '#FFD21C', '#FF8F07','#FF111B'],
        # color=[ '#97D6A7', '#A2DF44', '#FFD21C', '#FF8F07','#FF111B', '#8B0000'],
        label=["None", "Only one", "Both"],
        rwidth=0.8,
    )
    axs[1].set_title("Subsidence risk")
    axs[1].set_xlabel("Risk")
    axs[1].set_ylabel("Count")
    axs[1].set_ylim([0, 400])
    axs[1].set_xlim([-0.3, 1.1])
    axs[1].set_xticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    axs[1].set_xticklabels(
        ["No data", "Low", "Medium \n Low", "Medium", "Medium \n High", "High"]
    )  # Set the tick labels
    axs[1].legend()

    # Save the figure
    plt.savefig(
        data_path + r"/Plots" + r"/risk_histogram_SNTavail.jpg",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    # Create box plot for bridge type and Monitoring_class
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [
            bridges_gdf[bridges_gdf["Type"] == "Suspension"]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Type"] == "Arch"]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Type"] == "Cantilever"]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Type"] == "Cable Stayed"]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Type"] == "Truss"]["Monitoring_class"],
        ],
        labels=["Suspension", "Arch", "Cantilever", "Cable Stayed", "Truss"],
        patch_artist=True,
        boxprops=dict(facecolor="green", color="black"),
        whiskerprops=dict(color="black"),
        medianprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    plt.xlabel("Bridge type")
    plt.ylabel("Monitoring_class")
    plt.title("Box Plot: bridge type vs Monitoring_class")
    plt.savefig(
        os.path.join(data_path, "Plots", "type_vs_Monitoring_class_box.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Define the bridge types and colors
    bridge_types = ["Suspension", "Arch", "Cantilever", "Cable Stayed", "Truss"]
    colors = ["red", "green", "blue", "purple", "orange"]

    plt.figure(figsize=(10, 6))
    # Loop over each segment id
    for i in range(1, 6):
        # Create a list of data for each bridge type
        data = [
            bridges_gdf[(bridges_gdf["Type"] == bridge_type)][
                f"ps_count_100m_{i}"
            ].dropna()
            for bridge_type in bridge_types
        ]

        # Create a box plot for each bridge type
        for j, bridge_type in enumerate(bridge_types):
            plt.boxplot(
                data[j],
                positions=[i + j * 0.1 - 0.2],
                widths=0.07,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j], color=colors[j]),
            )

    # Create custom patches for the legend
    patches = [
        mpatches.Patch(color=colors[i], label=bridge_types[i])
        for i in range(len(bridge_types))
    ]

    plt.xticks(range(1, 6), range(1, 6))
    plt.xlabel("Segment id")
    plt.ylim([-5, 100])
    # plt.yscale('log')
    plt.ylabel("ps_count_100m")
    plt.title("Box Plot: Segment id vs ps_count_100m")
    plt.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Place the legend outside of the plot
    plt.savefig(
        os.path.join(data_path, "Plots", "segment_id_vs_ps_count_100m_box.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    # Loop over each segment id
    for i in range(1, 6):
        # Create a list of data for each bridge type
        data = [
            bridges_gdf[(bridges_gdf["Type"] == bridge_type)][
                f"ps_avail_perc_{i}"
            ].dropna()
            for bridge_type in bridge_types
        ]

        # Create a box plot for each bridge type
        for j, bridge_type in enumerate(bridge_types):
            plt.boxplot(
                data[j],
                positions=[i + j * 0.1 - 0.2],
                widths=0.07,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j], color=colors[j]),
            )

    # Create custom patches for the legend
    patches = [
        mpatches.Patch(color=colors[i], label=bridge_types[i])
        for i in range(len(bridge_types))
    ]

    plt.xticks(range(1, 6), range(1, 6))
    plt.xlabel("Segment id")
    # plt.ylim([-5,100])
    # plt.yscale('log')
    plt.ylabel("ps_avail_perc")
    plt.title("Box Plot: Segment id vs ps_avail_perc")
    plt.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Place the legend outside of the plot
    plt.savefig(
        os.path.join(data_path, "Plots", "segment_id_vs_ps_avail_perc_box.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create scatter plot for bridge length and Monitoring_class
    plt.figure(figsize=(10, 6))
    plt.scatter(
        bridges_gdf["Total Leng"],
        bridges_gdf["Monitoring_class"],
        color="green",
        alpha=0.5,
    )
    plt.xlabel("Total lenght of a bridge")
    plt.ylabel("Monitoring_class")
    plt.title("Scatter Plot: bridge total length vs Monitoring_class")
    plt.savefig(
        os.path.join(data_path, "Plots", "brdg_len_vs_Monitoring_class_scatter.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create box plot for bridge length and Monitoring_class
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [
            bridges_gdf[
                (bridges_gdf["Total Leng"] >= 0) & (bridges_gdf["Total Leng"] < 500)
            ]["Monitoring_class"],
            bridges_gdf[
                (bridges_gdf["Total Leng"] >= 500) & (bridges_gdf["Total Leng"] < 1000)
            ]["Monitoring_class"],
            bridges_gdf[
                (bridges_gdf["Total Leng"] >= 1000) & (bridges_gdf["Total Leng"] < 1500)
            ]["Monitoring_class"],
            bridges_gdf[
                (bridges_gdf["Total Leng"] >= 1500) & (bridges_gdf["Total Leng"] < 2000)
            ]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Total Leng"] >= 2000]["Monitoring_class"],
        ],
        labels=["0-500", "500-1000", "1000-1500", "1500-2000", ">2000"],
        patch_artist=True,
        boxprops=dict(facecolor="green", color="black"),
        whiskerprops=dict(color="black"),
        medianprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    plt.xlabel("Total length of a bridge")
    plt.ylabel("Monitoring_class")
    plt.title("Box Plot: Total length vs Monitoring_class")
    plt.savefig(
        os.path.join(data_path, "Plots", "total_length_vs_Monitoring_class_box.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create box plot for exposure and Monitoring_class
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [
            bridges_gdf[bridges_gdf["Exposure_norm"] == 0.2]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Exposure_norm"] == 0.4]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Exposure_norm"] == 0.6]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Exposure_norm"] == 0.8]["Monitoring_class"],
            bridges_gdf[bridges_gdf["Exposure_norm"] == 1]["Monitoring_class"],
        ],
        labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
        patch_artist=True,
        boxprops=dict(facecolor="green", color="black"),
        whiskerprops=dict(color="black"),
        medianprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    plt.xlabel("Exposure")
    plt.ylabel("Monitoring_class")
    plt.title("Box Plot: Exposure vs Monitoring_class")
    plt.savefig(
        os.path.join(data_path, "Plots", "exposure_vs_Monitoring_class_box.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create a dictionary to map bridge types to colors and markers
    bridge_types = bridges_gdf["Type"].unique()
    colors = ["blue", "orange", "green", "red", "purple"]
    markers = ["o", "v", "^", "<", ">"]
    type_dict = {
        bridge_type: (color, marker)
        for bridge_type, color, marker in zip(bridge_types, colors, markers)
    }

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for bridge_type, (color, marker) in type_dict.items():
        plt.scatter(
            bridges_gdf[bridges_gdf["Type"] == bridge_type]["Total Leng"],
            bridges_gdf[bridges_gdf["Type"] == bridge_type]["Monitoring_class"],
            color=color,
            marker=marker,
            alpha=0.5,
            label=bridge_type,
        )
    plt.xlabel("Total length of a bridge")
    plt.ylabel("Monitoring_class")
    plt.title("Bridge total length vs Monitoring_class")
    plt.xscale("log")  # Change x-axis to logarithmic scale
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Move legend outside of plot
    plt.savefig(
        os.path.join(
            data_path, "Plots", "brdg_len_vs_Monitoring_class_scatter_brdgtyp.jpg"
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create a dictionary to map bridge types to colors and markers
    bridge_types = bridges_gdf["Type"].unique()
    colors = ["blue", "orange", "green", "red", "purple"]
    markers = ["o", "v", "^", "<", ">"]
    type_dict = {
        bridge_type: (color, marker)
        for bridge_type, color, marker in zip(bridge_types, colors, markers)
    }

    # Create separate scatter plots for each bridge type
    for bridge_type, (color, marker) in type_dict.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(
            bridges_gdf[bridges_gdf["Type"] == bridge_type]["Total Leng"],
            bridges_gdf[bridges_gdf["Type"] == bridge_type]["Monitoring_class"],
            color=color,
            marker=marker,
            alpha=0.5,
            label=bridge_type,
        )
        plt.xlabel("Total length of a bridge")
        plt.ylabel("Monitoring_class")
        plt.title(f"Bridge total length vs Monitoring_class for {bridge_type}")
        plt.xscale("log")  # Change x-axis to logarithmic scale
        plt.legend(
            bbox_to_anchor=(0.5, -0.15), loc="lower center"
        )  # Move legend to the bottom
        plt.savefig(
            os.path.join(
                data_path,
                "Plots",
                f"brdg_len_vs_Monitoring_class_scatter_{bridge_type}.jpg",
            ),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create histogram of monitoring class

    # Define bin edges
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # Make the plot
    plt.figure(figsize=(10, 6))
    weights = np.ones_like(bridges_gdf["Monitoring_class"]) / len(
        bridges_gdf["Monitoring_class"]
    )
    plt.hist(
        bridges_gdf["Monitoring_class"],
        bins=bin_edges,
        edgecolor="black",
        weights=weights,
        rwidth=0.7,
    )
    plt.xlabel("Monitoring_class")
    plt.ylabel("Percentage")
    plt.title("Histogram of Monitoring_class")
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
    )  # Format y-axis as percentages
    plt.savefig(
        os.path.join(data_path, "Plots", "Monitoring_class_histogram.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Create histogram of monitoring class per segment

    # Define bin edges
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for i in range(1, 6):
        plt.figure(figsize=(10, 6))
        weights = np.ones_like(bridges_gdf[f"Monitoring_class_{i}"]) / len(
            bridges_gdf[f"Monitoring_class_{i}"]
        )
        plt.hist(
            bridges_gdf[f"Monitoring_class_{i}"],
            bins=bin_edges,
            edgecolor="black",
            weights=weights,
            rwidth=0.7,
        )
        plt.xlabel(f"Monitoring_class_{i}")
        plt.ylabel("Percentage")
        plt.title(f"Histogram of Monitoring_class_{i}")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
        )  # Format y-axis as percentages
        plt.ylim(0, 0.6)
        plt.savefig(
            os.path.join(data_path, "Plots", f"Monitoring_class_histogram_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Define the values
    values = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for i in range(1, 6):
        plt.figure(figsize=(10, 6))

        # Count the occurrences of each value
        counts = (
            bridges_gdf[f"Monitoring_class_{i}"]
            .value_counts(normalize=True)
            .sort_index()
        )

        # Create a bar plot of the counts
        counts.plot.bar(edgecolor="black", width=0.7)

        plt.xlabel(f"Monitoring_class_{i}")
        plt.ylabel("Percentage")
        plt.title(f"Histogram of Monitoring_class_{i}")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
        )  # Format y-axis as percentages
        plt.ylim(0, 0.6)
        plt.xticks(
            range(len(values)), values, rotation=0
        )  # Set x-ticks to be the values
        plt.savefig(
            os.path.join(data_path, "Plots", f"Monitoring_class_histogram_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Define bin edges
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100]

    # Create bar plot of ps_mean per segment
    for i in range(1, 6):
        plt.figure(figsize=(10, 6))
        data = bridges_gdf[f"ps_mean_world_{i}"]
        counts, bin_edges = np.histogram(data, bins=bin_edges)
        weights = counts / len(data)
        bin_widths = 0.4
        bin_centers = bin_edges[:-1] + bin_widths / 2
        plt.bar(
            bin_centers, weights, width=bin_widths, edgecolor="black", align="center"
        )
        plt.xlabel(f"ps_mean_world_{i}")
        plt.ylabel("Percentage")
        plt.ylim(0, 0.35)
        plt.title(f"Bar plot of ps_mean_world_{i}")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
        )  # Format y-axis as percentages
        plt.xticks(bin_edges[:-1])  # Set x-ticks at the bin edges
        plt.savefig(
            os.path.join(data_path, "Plots", f"ps_mean_barplot_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create histogram of ps count per 100m per segment
    # Define bin edges
    bin_edges = [0, 1, 3, 5, 10, 10000]

    for i in range(1, 6):
        plt.figure(figsize=(10, 6))
        data = bridges_gdf[f"ps_count_100m_{i}"]
        counts, bin_edges = np.histogram(data, bins=bin_edges)
        weights = counts / len(data)
        bin_widths = 0.4
        bin_centers = bin_edges[:-1] + bin_widths / 2
        plt.bar(
            bin_centers, weights, width=bin_widths, edgecolor="black", align="center"
        )
        plt.xlabel(f"ps_count_100m_{i}")
        plt.ylabel("Percentage")
        plt.ylim(0, 0.5)
        plt.title(f"Bar plot of ps_count_100m_{i}")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
        )  # Format y-axis as percentages
        plt.xticks(bin_edges[:-1])  # Set x-ticks at the bin edges
        plt.savefig(
            os.path.join(data_path, "Plots", f"ps_count_100m_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create histogram of ps percent avail per segment
    # Define bin edges
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for i in range(1, 6):
        plt.figure(figsize=(10, 6))
        data = bridges_gdf[f"ps_avail_perc_{i}"]
        counts, bin_edges = np.histogram(data, bins=bin_edges)
        weights = counts / len(data)
        bin_widths = 0.2
        bin_centers = bin_edges[:-1] + bin_widths / 2
        plt.bar(
            bin_centers, weights, width=bin_widths, edgecolor="black", align="center"
        )
        plt.xlabel(f"ps_avail_perc_{i}")
        plt.ylabel("Percentage")
        plt.ylim(0, 0.6)
        plt.title(f"Bar plot of ps_avail_perc_{i}")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
        )  # Format y-axis as percentages
        plt.xticks(bin_edges[:-1])  # Set x-ticks at the bin edges
        plt.savefig(
            os.path.join(data_path, "Plots", f"ps_avail_perc_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create a scatter plot of PS availability and the count per 100 m for each segment
    for i in range(1, 6):
        plt.figure(figsize=(10, 6))

        for edge in [1, 3, 5, 10]:
            plt.axvline(
                x=edge, color="r", linestyle="--", alpha=0.2
            )  # Draw vertical line at each bin edge

        for edge in [0.2, 0.4, 0.6, 0.8]:
            plt.axhline(
                y=edge, color="b", linestyle="--", alpha=0.2
            )  # Draw vertical line at each bin edge

        plt.scatter(
            bridges_gdf[f"ps_count_100m_{i}"],
            bridges_gdf[f"ps_avail_perc_{i}"],
            color="green",
            alpha=0.5,
            s=5,
        )
        plt.xlabel("Count of PS per 100 m")
        plt.ylabel("PS availability in percentages")
        plt.title(f"PS availability vs ps count for segment {i}")
        plt.xscale("log")  # Change x-axis to logarithmic scale

        plt.gca().set_xticks([1, 3, 5, 10, 100, 200])  # Set x-axis ticks
        plt.gca().get_xaxis().set_major_formatter(
            mticker.ScalarFormatter()
        )  # Format x-axis ticks as scalar

        plt.xlim(0, 300)  # Set x-axis limits
        # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center')  # Move legend to the bottom
        plt.savefig(
            os.path.join(data_path, "Plots", f"PSavail_vs_PScount_scatter_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create a scatter plot of PS availability and the length of each segment
    for i in range(1, 6):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            bridges_gdf[f"length_{i}"],
            bridges_gdf[f"ps_avail_perc_{i}"],
            color="green",
            alpha=0.5,
        )
        plt.xlabel("Lenght of segment in m")
        plt.ylabel("PS availability in percentages")
        plt.title(f"PS availability vs length for segment {i}")
        plt.xscale("log")  # Change x-axis to logarithmic scale

        # for edge in [1,3,5,10]:
        #     plt.axvline(x=edge, color='r', linestyle='--')  # Draw vertical line at each bin edge

        # for edge in [0.2, 0.4, 0.6, 0.8]:
        #     plt.axhline(y=edge, color='b', linestyle='--')  # Draw vertical line at each bin edge

        # plt.gca().set_xticks([1, 3, 5, 10, 100, 200])  # Set x-axis ticks
        plt.gca().get_xaxis().set_major_formatter(
            mticker.ScalarFormatter()
        )  # Format x-axis ticks as scalar

        # plt.xlim(0, 300)  # Set x-axis limits
        # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center')  # Move legend to the bottom
        plt.savefig(
            os.path.join(data_path, "Plots", f"PSavail_vs_length_scatter_{i}.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    # Create a scatter plot of PS availability and the count per 100 m for each segment
    colors = ["blue", "green", "red", "cyan", "magenta"]  # Define a list of colors
    plt.figure(figsize=(10, 6))

    for i in range(1, 6):
        plt.scatter(
            bridges_gdf[f"ps_count_100m_{i}"],
            bridges_gdf[f"ps_avail_perc_{i}"],
            color=colors[i - 1],
            alpha=0.5,
            s=10,
            label=f"Segment {i}",
        )
        plt.xlabel("Count of PS per 100 m")
        plt.ylabel("PS availability in percentages")
        plt.title("PS availability vs ps count for all segments")
        plt.xscale("log")  # Change x-axis to logarithmic scale

        for edge in [1, 3, 5, 10]:
            plt.axvline(
                x=edge, color="r", linestyle="--"
            )  # Draw vertical line at each bin edge

        for edge in [0.2, 0.4, 0.6, 0.8]:
            plt.axhline(
                y=edge, color="b", linestyle="--"
            )  # Draw horizontal line at each bin edge

        plt.gca().set_xticks([1, 3, 5, 10, 100, 200])  # Set x-axis ticks
        plt.gca().get_xaxis().set_major_formatter(
            mticker.ScalarFormatter()
        )  # Format x-axis ticks as scalar

    plt.xlim(0, 300)  # Set x-axis limits
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Move legend to the outside of the plot
    plt.savefig(
        os.path.join(data_path, "Plots", "PSavail_vs_PScount_scatter_combined.jpg"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Select columns that start with 'monitoring_clas' and add 'ID' column
    lsb_gdf_filtered = gpd.GeoDataFrame(
        bridges_gdf.filter(regex="^Monitoring_class").assign(
            ID=bridges_gdf["ID"], geometry=bridges_gdf["geometry"]
        )
    )
    lsb_gdf_filtered = lsb_gdf_filtered.drop(
        ["Monitoring_class_mean", "Monitoring_class_min", "Monitoring_class_bins"],
        axis=1,
    )
    lsb_gdf_filtered = lsb_gdf_filtered.rename(
        columns=lambda x: (
            x.replace("Monitoring_class", "Mnt_cl")
            if x.startswith("Monitoring_class")
            else x
        )
    )

    # Export GeoDataFrame to shapefile
    lsb_gdf_filtered.to_file(os.path.join(data_path, "monitoring_class.shp"))

    # Create a dictionary to map bridge types to colors and markers
    bridge_types = bridges_gdf["Type"].unique()
    colors = ["blue", "orange", "green", "red", "purple"]
    markers = ["o", "v", "^", "<", ">"]
    type_dict = {
        bridge_type: (color, marker)
        for bridge_type, color, marker in zip(bridge_types, colors, markers)
    }

    # Create scatter plot
    for i in range(1, 6):

        plt.figure(figsize=(10, 6))
        for bridge_type, (color, marker) in type_dict.items():
            plt.scatter(
                bridges_gdf[bridges_gdf["Type"] == bridge_type][f"ps_count_100m_{i}"],
                bridges_gdf[bridges_gdf["Type"] == bridge_type][f"ps_avail_perc_{i}"],
                color=color,
                marker=marker,
                alpha=0.5,
                label=bridge_type,
            )
        plt.xlabel("Count of PS per 100 m")
        plt.ylabel("PS availability in percentages")
        plt.title(f"PS availability vs ps count for segment {i}")
        plt.xscale("log")  # Change x-axis to logarithmic scale
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left"
        )  # Move legend outside of plot
        plt.savefig(
            os.path.join(
                data_path, "Plots", f"PSavail_vs_PScount_scatter_{i}_brdgtyp.jpg"
            ),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    print("end")
