"""Analyse slope distribution in the WSL inventory & combined with Storme."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl

SLOPE_RASTER = Path("data/swissalti_slope/slope_deg_25m_ch.tif")
SLOPE_FILENAME = "slope_distribution"
OUTDIR = Path("output/data_analysis/07_wsl_inventory_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


def slope_analysis(df, dataset_name="WSL Inventory"):
    """Calculate and plot the overall slope distribution."""

    slope = pd.to_numeric(
        df["slope_from_map"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).dropna()

    summary = (
        f"n = {len(slope)}\n"
        f"median = {slope.median():.1f}°\n"
        f"mean = {slope.mean():.1f}°\n"
        f"std = {slope.std():.1f}°\n"
    )

    with open(
        f"{OUTDIR}/{SLOPE_FILENAME}_{dataset_name.replace(' ', '_')}_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    plt.figure(figsize=(8, 5))
    plt.hist(
        slope,
        bins=np.linspace(0, 51, 50),
        color="lightblue",
        edgecolor="black",
    )
    plt.xlabel("Slope angle [°]")
    plt.ylabel("Count")
    plt.title(f"Slope distribution of {dataset_name}")
    plt.tight_layout()
    plt.savefig(
        f"{OUTDIR}/{SLOPE_FILENAME}_{dataset_name.replace(' ', '_')}_histogram.png",
        dpi=150,
    )
    plt.close()


def get_slope_from_map(df: pd.DataFrame) -> pd.DataFrame:
    """Extract raster slope values at WSL inventory point coordinates."""

    required_columns = {"x", "y"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required coordinate columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    result = df.copy()

    coordinates = list(
        zip(
            pd.to_numeric(result["x"], errors="coerce"),
            pd.to_numeric(result["y"], errors="coerce"),
        )
    )

    with rasterio.open(SLOPE_RASTER) as source:
        slope_values = []

        for x, y in coordinates:
            if pd.isna(x) or pd.isna(y):
                slope_values.append(float("nan"))
                continue

            if not (
                source.bounds.left <= x <= source.bounds.right
                and source.bounds.bottom <= y <= source.bounds.top
            ):
                slope_values.append(float("nan"))
                continue

            value = next(source.sample([(x, y)]))[0]

            if source.nodata is not None and value == source.nodata:
                slope_values.append(float("nan"))
            else:
                slope_values.append(float(value))

    result["slope_from_map"] = slope_values
    return result


def main() -> None:
    """Run the WSL inventory slope analysis."""

    if not SLOPE_RASTER.exists():
        raise FileNotFoundError(f"Slope raster not found: {SLOPE_RASTER}")

    dfs = {
        "WSL Inventory": dl.load_wsl_usable_inventory(),
        "Storme Inventory": dl.load_storme_inventory(),
        "Combined Inventory": dl.load_combined_inventory(),
    }

    for dataset_name, df in dfs.items():
        if df.empty:
            raise ValueError(f"The {dataset_name} is empty.")

        print(f"Analyzing {len(df)} events in {dataset_name}...")
        df = get_slope_from_map(df)
        slope_analysis(df, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
