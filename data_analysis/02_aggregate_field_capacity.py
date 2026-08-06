"""Aggregate the BAFU field capacity data by year and month, and save the results to CSV files."""

from itertools import cycle
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import region_map as rm

INPUT_FILE = "data/soil_moisture_history/weekly_historic_regions.csv"
OUTDIR = "output/data_analysis/02_aggregate_moisture"
YEARLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_yearly.csv"
MONTHLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_monthly.csv"
os.makedirs(OUTDIR, exist_ok=True)


def animate_monthly_moisture(mean_by_region, output_file):
    """
    Animate mean monthly saturation per region as a Swiss choropleth GIF.

    mean_by_region: long DataFrame with columns
        [drought_region_id, month, saturation_ratio].
    Region borders are drawn as thin lines and each region ID is written inside.
    """
    ids = sorted(mean_by_region["drought_region_id"].unique())
    geometries, label_points, _ = dl.load_regions(ids)

    rm.animate_monthly_regions(
        mean_by_region,
        geometries,
        region_col="drought_region_id",
        month_col="month",
        value_col="saturation_ratio",
        label_points=label_points,
        output_file=output_file,
        fps=2,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        title="Mean monthly field capacity",
        cbar_label="Saturation ratio (nFK)",
    )


def aggregate_moisture_data():
    """Aggregate the BAFU field capacity data by year and month, and save the results to CSV files."""
    # Load data
    df = pd.read_csv(
        INPUT_FILE, sep=",", skiprows=3, parse_dates=["measured_at"], dayfirst=True
    )
    df["saturation_ratio"] = df["soil_moisture_ufc"] / 100.0

    # 1. Yearly Aggregation
    yearly = (
        df.groupby(["drought_region_id", df["measured_at"].dt.year])["saturation_ratio"]
        .mean()
        .reset_index()
    )
    yearly.columns = ["region_id", "year", "mean_moisture"]
    yearly.to_csv(YEARLY_OUTPUT_FILE, index=False)

    # 2. Monthly Aggregation
    df["year_month"] = df["measured_at"].dt.to_period("M")
    monthly = (
        df.groupby(["drought_region_id", "year_month"])["saturation_ratio"]
        .mean()
        .reset_index()
    )
    monthly.to_csv(MONTHLY_OUTPUT_FILE, index=False)

    # 3. Mean Monthly Moisture by Region
    df["month"] = df["measured_at"].dt.month
    mean_monthly_by_region = (
        df.groupby(["drought_region_id", "month"])["saturation_ratio"]
        .mean()
        .reset_index()
    )
    mean_monthly_by_region.to_csv(f"{OUTDIR}/mean_moisture_by_region.csv", index=False)

    # Plot the mean monthly moisture by region
    plt.figure(figsize=(12, 6))
    markers = ["o", "s", "^", "D"]

    for i, region_id in enumerate(mean_monthly_by_region["drought_region_id"].unique()):
        region_data = mean_monthly_by_region[
            mean_monthly_by_region["drought_region_id"] == region_id
        ]
        marker_custom = markers[(i // 10) % len(markers)]

        plt.plot(
            region_data["month"],
            region_data["saturation_ratio"],
            marker=marker_custom,
            label=f"{region_id}",
        )
    plt.title("Mean Monthly Moisture by Region")
    plt.xlabel("Month")
    plt.ylabel("Mean Saturation Ratio")
    plt.xticks(range(1, 13))

    # Position legend horizontally below the plot
    plt.legend(
        title="Region",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # outside the axes on the right
        ncol=2,  # horizontal layout
        fontsize="small",
        frameon=False,
    )

    # Leave room for the legend
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTDIR}/mean_monthly_moisture_by_region.png")

    # 4. Animate the mean monthly moisture by region
    animate_monthly_moisture(
        mean_monthly_by_region, f"{OUTDIR}/mean_monthly_moisture.gif"
    )

    print("Aggregation complete. Results saved to CSV files.")


if __name__ == "__main__":
    aggregate_moisture_data()
