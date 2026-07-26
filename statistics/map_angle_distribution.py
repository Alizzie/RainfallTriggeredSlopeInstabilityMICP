"""
Analyse slope-angle proportions and historical mean annual rainfall.

Inputs
------
Slope raster:
    data/swissalti_slope/slope_deg_25m_ch.tif

Rainfall:
    Loaded through core.data_loader.load_rainfall_grid()

Outputs
-------
Slope:
    slope_angle_distribution.csv
    slope_angle_classes.tif
    slope_angle_classes.png

Rainfall:
    mean_annual_rainfall.tif
    mean_annual_rainfall.png
    rainfall_by_slope_class.csv
    rainfall_by_slope_class.png
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

from core import data_loader

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SLOPE_RASTER = Path("data/swissalti_slope/slope_deg_25m_ch.tif")

OUTPUT_DIR = Path("output/statistics/slope_rainfall_analysis")

# Adjust this range to the years available in data/rhiresD.
RAINFALL_YEARS = range(1991, 2025)

SLOPE_BINS = np.array(
    [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        90,
    ],
    dtype=float,
)

SLOPE_LABELS = [
    "0-<5°",
    "5-<10°",
    "10-<15°",
    "15-<20°",
    "20-<25°",
    "25-<30°",
    "30-<35°",
    "35-<40°",
    "40-<45°",
    "45-90°",
]

SLOPE_COLORS = [
    "#ffffcc",
    "#d9f0a3",
    "#addd8e",
    "#78c679",
    "#41ab5d",
    "#238443",
    "#2c7fb8",
    "#225ea8",
    "#253494",
    "#54278f",
]


# ---------------------------------------------------------------------
# Slope analysis
# ---------------------------------------------------------------------


def classify_slope(
    slope: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert continuous slope angles to categorical classes.

    Class IDs begin at 1. Zero represents NoData.
    """

    classes = np.zeros(
        slope.shape,
        dtype=np.uint8,
    )

    classified = np.digitize(
        slope,
        bins=SLOPE_BINS,
        right=False,
    )

    # Include exactly 90° in the final class.
    classified[np.isclose(slope, SLOPE_BINS[-1])] = len(SLOPE_LABELS)

    valid = (
        valid_mask
        & np.isfinite(slope)
        & (slope >= SLOPE_BINS[0])
        & (slope <= SLOPE_BINS[-1])
    )

    classes[valid] = classified[valid]

    return classes


def calculate_slope_distribution(
    slope_classes: np.ndarray,
    pixel_area_m2: float,
) -> pd.DataFrame:
    """Calculate area and proportion for every slope class."""

    total_valid_cells = np.count_nonzero(slope_classes > 0)

    rows = []

    for class_id, label in enumerate(
        SLOPE_LABELS,
        start=1,
    ):
        cell_count = int(np.count_nonzero(slope_classes == class_id))

        area_m2 = cell_count * pixel_area_m2

        proportion = cell_count / total_valid_cells if total_valid_cells > 0 else np.nan

        rows.append(
            {
                "class_id": class_id,
                "slope_class": label,
                "lower_angle_deg": (SLOPE_BINS[class_id - 1]),
                "upper_angle_deg": (SLOPE_BINS[class_id]),
                "cell_count": cell_count,
                "area_km2": area_m2 / 1_000_000,
                "proportion": proportion,
                "proportion_percent": (proportion * 100),
            }
        )

    return pd.DataFrame(rows)


def create_slope_colormap() -> ListedColormap:
    """Create the categorical slope colormap."""

    colormap = ListedColormap(SLOPE_COLORS)
    colormap.set_bad("white")

    return colormap


def raster_extent(
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> list[float]:
    """Return plotting extent for a raster."""

    height, width = shape

    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height

    return [
        left,
        right,
        bottom,
        top,
    ]


def write_slope_class_raster(
    slope_classes: np.ndarray,
    source_profile: dict,
    output_file: Path,
) -> None:
    """Write categorized slope classes as GeoTIFF."""

    profile = source_profile.copy()

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as destination:
        destination.write(
            slope_classes,
            1,
        )


def plot_slope_classes(
    slope_classes: np.ndarray,
    transform: rasterio.Affine,
    output_file: Path,
) -> None:
    """Create a colored slope-class image."""

    masked = np.ma.masked_where(
        slope_classes == 0,
        slope_classes,
    )

    colormap = create_slope_colormap()

    boundaries = np.arange(
        0.5,
        len(SLOPE_LABELS) + 1.5,
        1,
    )

    norm = BoundaryNorm(
        boundaries,
        colormap.N,
    )

    fig, ax = plt.subplots(figsize=(11, 8))

    image = ax.imshow(
        masked,
        cmap=colormap,
        norm=norm,
        extent=raster_extent(
            transform,
            slope_classes.shape,
        ),
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=np.arange(
            1,
            len(SLOPE_LABELS) + 1,
        ),
        fraction=0.035,
        pad=0.02,
    )

    colorbar.ax.set_yticklabels(SLOPE_LABELS)
    colorbar.set_label("Slope-angle class")

    ax.set_title("Slope-angle classes from swissALTI3D")
    ax.set_xlabel("Easting LV95")
    ax.set_ylabel("Northing LV95")
    ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def analyse_slope():
    """Load and analyse the slope raster."""

    if not SLOPE_RASTER.exists():
        raise FileNotFoundError(f"Slope raster missing: {SLOPE_RASTER}")

    with rasterio.open(SLOPE_RASTER) as source:
        slope = source.read(1).astype(np.float32)

        valid_mask = source.read_masks(1) > 0

        if source.nodata is not None:
            valid_mask &= ~np.isclose(
                slope,
                source.nodata,
            )

        profile = source.profile.copy()
        transform = source.transform
        crs = source.crs

    pixel_area_m2 = abs(transform.a) * abs(transform.e)

    slope_classes = classify_slope(
        slope=slope,
        valid_mask=valid_mask,
    )

    distribution = calculate_slope_distribution(
        slope_classes=slope_classes,
        pixel_area_m2=pixel_area_m2,
    )

    distribution.to_csv(
        OUTPUT_DIR / "slope_angle_distribution.csv",
        index=False,
        float_format="%.6f",
    )

    write_slope_class_raster(
        slope_classes=slope_classes,
        source_profile=profile,
        output_file=(OUTPUT_DIR / "slope_angle_classes.tif"),
    )

    plot_slope_classes(
        slope_classes=slope_classes,
        transform=transform,
        output_file=(OUTPUT_DIR / "slope_angle_classes.png"),
    )

    return (
        slope_classes,
        transform,
        crs,
        distribution,
    )


# ---------------------------------------------------------------------
# Rainfall analysis
# ---------------------------------------------------------------------


def calculate_mean_annual_rainfall(
    rainfall,
):
    """
    Sum daily rainfall within each year and average the annual totals.

    The result is mean annual rainfall in mm/year when the daily
    RhiresD values are expressed in mm/day.
    """

    rainfall = rainfall.where(np.isfinite(rainfall))

    annual_totals = rainfall.resample(time="YS").sum(
        dim="time",
        skipna=True,
        min_count=1,
    )

    mean_annual = annual_totals.mean(
        dim="time",
        skipna=True,
    )

    mean_annual.name = "mean_annual_rainfall"

    return mean_annual


def rainfall_to_raster(
    rainfall,
) -> tuple[
    np.ndarray,
    rasterio.Affine,
]:
    """
    Convert the RhiresD DataArray into a north-up raster.

    Expected spatial coordinates:
        E = Easting
        N = Northing
    """

    if "E" not in rainfall.dims:
        raise ValueError(
            "Rainfall data have no E dimension. " f"Dimensions: {rainfall.dims}"
        )

    if "N" not in rainfall.dims:
        raise ValueError(
            "Rainfall data have no N dimension. " f"Dimensions: {rainfall.dims}"
        )

    rainfall = rainfall.transpose(
        "N",
        "E",
    )

    eastings = np.asarray(
        rainfall["E"].values,
        dtype=float,
    )

    northings = np.asarray(
        rainfall["N"].values,
        dtype=float,
    )

    values = np.asarray(
        rainfall.values,
        dtype=np.float32,
    )

    # Ensure west-to-east column order.
    if eastings[0] > eastings[-1]:
        eastings = eastings[::-1]
        values = np.fliplr(values)

    # Ensure north-to-south row order.
    if northings[0] < northings[-1]:
        northings = northings[::-1]
        values = np.flipud(values)

    x_resolution = abs(float(np.median(np.diff(eastings))))

    y_resolution = abs(float(np.median(np.diff(northings))))

    west = eastings[0] - x_resolution / 2
    north = northings[0] + y_resolution / 2

    transform = from_origin(
        west,
        north,
        x_resolution,
        y_resolution,
    )

    return values, transform


def write_rainfall_raster(
    rainfall: np.ndarray,
    transform: rasterio.Affine,
    output_file: Path,
) -> None:
    """Write mean annual rainfall as an LV95 GeoTIFF."""

    nodata = -9999.0

    output = np.where(
        np.isfinite(rainfall),
        rainfall,
        nodata,
    ).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:2056",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }

    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as destination:
        destination.write(
            output,
            1,
        )


def plot_mean_annual_rainfall(
    rainfall: np.ndarray,
    transform: rasterio.Affine,
    output_file: Path,
) -> None:
    """Create a map of mean annual rainfall."""

    masked = np.ma.masked_invalid(rainfall)

    fig, ax = plt.subplots(figsize=(11, 8))

    image = ax.imshow(
        masked,
        cmap="Blues",
        extent=raster_extent(
            transform,
            rainfall.shape,
        ),
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.02,
    )

    colorbar.set_label("Mean annual rainfall (mm/year)")

    ax.set_title("Mean annual RhiresD rainfall")
    ax.set_xlabel("Easting LV95")
    ax.set_ylabel("Northing LV95")
    ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def reproject_slope_classes(
    slope_classes: np.ndarray,
    slope_transform: rasterio.Affine,
    slope_crs,
    rainfall_shape: tuple[int, int],
    rainfall_transform: rasterio.Affine,
) -> np.ndarray:
    """
    Reproject slope classes to the RhiresD grid.

    Nearest-neighbour interpolation preserves categorical class IDs.
    """

    destination = np.zeros(
        rainfall_shape,
        dtype=np.uint8,
    )

    reproject(
        source=slope_classes,
        destination=destination,
        src_transform=slope_transform,
        src_crs=slope_crs,
        src_nodata=0,
        dst_transform=rainfall_transform,
        dst_crs="EPSG:2056",
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return destination


def calculate_rainfall_by_slope_class(
    mean_annual_rainfall: np.ndarray,
    slope_classes: np.ndarray,
) -> pd.DataFrame:
    """Calculate rainfall statistics for every slope class."""

    rows = []

    for class_id, label in enumerate(
        SLOPE_LABELS,
        start=1,
    ):
        mask = (slope_classes == class_id) & np.isfinite(mean_annual_rainfall)

        values = mean_annual_rainfall[mask]

        if values.size == 0:
            mean_value = np.nan
            median_value = np.nan
            standard_deviation = np.nan
            minimum = np.nan
            maximum = np.nan
        else:
            mean_value = float(np.nanmean(values))
            median_value = float(np.nanmedian(values))
            standard_deviation = float(np.nanstd(values))
            minimum = float(np.nanmin(values))
            maximum = float(np.nanmax(values))

        rows.append(
            {
                "class_id": class_id,
                "slope_class": label,
                "lower_angle_deg": (SLOPE_BINS[class_id - 1]),
                "upper_angle_deg": (SLOPE_BINS[class_id]),
                "rainfall_cell_count": (int(values.size)),
                "mean_annual_rainfall_mm": (mean_value),
                "median_annual_rainfall_mm": (median_value),
                "std_annual_rainfall_mm": (standard_deviation),
                "min_annual_rainfall_mm": (minimum),
                "max_annual_rainfall_mm": (maximum),
            }
        )

    return pd.DataFrame(rows)


def plot_rainfall_by_slope_class(
    table: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot mean annual rainfall for every slope class."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        table["slope_class"],
        table["mean_annual_rainfall_mm"],
    )

    ax.set_xlabel("Slope-angle class")
    ax.set_ylabel("Mean annual rainfall (mm/year)")
    ax.set_title("Mean annual rainfall by slope-angle class")

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def analyse_rainfall(
    slope_classes: np.ndarray,
    slope_transform: rasterio.Affine,
    slope_crs,
) -> pd.DataFrame:
    """Load and analyse historical gridded rainfall."""

    rainfall = data_loader.load_rainfall_grid(
        years=RAINFALL_YEARS,
        # A monthly time chunk can reduce memory usage when dask
        # is available. Set chunks=None if this causes problems.
        chunks={"time": 31},
    )

    if rainfall is None:
        raise FileNotFoundError(
            "No rainfall files were loaded for the " "configured year range."
        )

    try:
        mean_annual = calculate_mean_annual_rainfall(rainfall)

        rainfall_array, rainfall_transform = rainfall_to_raster(mean_annual)

        # Trigger computation before closing NetCDF datasets.
        rainfall_array = np.asarray(
            rainfall_array,
            dtype=np.float32,
        )

    finally:
        rainfall.close()

    write_rainfall_raster(
        rainfall=rainfall_array,
        transform=rainfall_transform,
        output_file=(OUTPUT_DIR / "mean_annual_rainfall.tif"),
    )

    plot_mean_annual_rainfall(
        rainfall=rainfall_array,
        transform=rainfall_transform,
        output_file=(OUTPUT_DIR / "mean_annual_rainfall.png"),
    )

    slope_on_rainfall_grid = reproject_slope_classes(
        slope_classes=slope_classes,
        slope_transform=slope_transform,
        slope_crs=slope_crs,
        rainfall_shape=rainfall_array.shape,
        rainfall_transform=rainfall_transform,
    )

    table = calculate_rainfall_by_slope_class(
        mean_annual_rainfall=rainfall_array,
        slope_classes=slope_on_rainfall_grid,
    )

    table.to_csv(
        OUTPUT_DIR / "rainfall_by_slope_class.csv",
        index=False,
        float_format="%.3f",
    )

    plot_rainfall_by_slope_class(
        table=table,
        output_file=(OUTPUT_DIR / "rainfall_by_slope_class.png"),
    )

    return table


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Run slope and rainfall analyses."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        slope_classes,
        slope_transform,
        slope_crs,
        slope_distribution,
    ) = analyse_slope()

    print("\nSlope-angle distribution:")

    print(
        slope_distribution[
            [
                "slope_class",
                "area_km2",
                "proportion_percent",
            ]
        ].to_string(index=False)
    )

    rainfall_table = analyse_rainfall(
        slope_classes=slope_classes,
        slope_transform=slope_transform,
        slope_crs=slope_crs,
    )

    print("\nMean annual rainfall by slope class:")

    print(
        rainfall_table[
            [
                "slope_class",
                "rainfall_cell_count",
                "mean_annual_rainfall_mm",
                "median_annual_rainfall_mm",
            ]
        ].to_string(index=False)
    )

    print("\nResults written to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
