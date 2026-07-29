"""Aggregate the BAFU field capacity data by year and month, and save the results to CSV files."""

import os
import pandas as pd

INPUT_FILE = "data/soil_moisture_history/weekly_historic_regions.csv"
OUTDIR = "output/statistics/01_aggregate_moisture"
YEARLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_yearly.csv"
MONTHLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_monthly.csv"
os.makedirs(OUTDIR, exist_ok=True)


def aggregate_moisture_data():
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

    print(
        "Aggregation complete: 'mean_moisture_yearly.csv' and 'mean_moisture_monthly.csv' created."
    )


if __name__ == "__main__":
    aggregate_moisture_data()
