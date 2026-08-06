"""
Analyse slope-angle proportions and historical mean annual rainfall.
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
from core import region_map

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SLOPE_RASTER = Path("data/swissalti_slope/slope_deg_25m_ch.tif")

OUTPUT_DIR = Path("output/data_analysis/04_slope_rainfall_analysis")

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
# Region helpers (shared by slope + rainfall)
# ---------------------------------------------------------------------


def slope_stats_by_region(
    slope: np.ndarray,
    valid_mask: np.ndarray,
    slope_transform: rasterio.Affine,
    region_ids: list,
) -> pd.DataFrame:
    """
    Per-region slope-angle statistics (min, median, mean, max, count).

    Drought polygons are rasterised onto the slope grid, then the slope values
    inside each region are summarised.
    """
    region_raster = data_loader.rasterize_drought_regions(
        rainfall_shape=slope.shape,
        rainfall_transform=slope_transform,
        required_region_ids=region_ids,
    )

    usable = valid_mask & np.isfinite(slope)
    rows = []
    for region_id in region_ids:
        mask = usable & (region_raster == region_id)
        values = slope[mask]
        if values.size == 0:
            rows.append(
                {
                    "region_id": region_id,
                    "cell_count": 0,
                    "min_slope_deg": np.nan,
                    "median_slope_deg": np.nan,
                    "mean_slope_deg": np.nan,
                    "max_slope_deg": np.nan,
                }
            )
            continue
        rows.append(
            {
                "region_id": region_id,
                "cell_count": int(values.size),
                "min_slope_deg": float(np.min(values)),
                "median_slope_deg": float(np.median(values)),
                "mean_slope_deg": float(np.mean(values)),
                "max_slope_deg": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Slope analysis
# ---------------------------------------------------------------------


def plot_slope_histogram(
    valid_slopes: np.ndarray,
    output_file: Path,
) -> None:
    """Plot a histogram of each slope angles."""

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 90, 91)

    ax.hist(
        valid_slopes,
        bins=bins,
        color="lightblue",
        edgecolor="black",
    )

    ax.set_xlabel("Slope angle [°]")
    ax.set_ylabel("Frequency (pixel count)")
    ax.set_title("Slope-angle distribution")

    ax.set_xlim(0, 90)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


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
    geometries: dict = None,
    label_points: dict = None,
    axis_off: bool = False,
) -> None:
    """
    Create a colored slope-class image.

    If ``geometries`` is given, region borders are drawn as thin lines with the
    region ID inside each one. ``axis_off`` hides the LV95 coordinates so only
    the map and the class legend remain.
    """

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

    if geometries:
        region_map.draw_region_borders(
            ax,
            geometries,
            label_points=label_points,
            label_ids=True,
            border_kw={"edgecolor": "black", "linewidth": 0.4},
        )

    ax.set_title("Slope-angle classes from swissALTI3D")
    ax.set_aspect("equal")

    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel("Easting LV95")
        ax.set_ylabel("Northing LV95")

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
        axis_off=True,
    )

    # Per-region slope statistics + a map with region borders and IDs (no axes).
    geometries, label_points, region_ids = data_loader.load_regions()

    region_stats = slope_stats_by_region(
        slope=slope,
        valid_mask=valid_mask,
        slope_transform=transform,
        region_ids=region_ids,
    )
    region_stats.to_csv(
        OUTPUT_DIR / "slope_stats_by_region.csv",
        index=False,
        float_format="%.3f",
    )

    plot_slope_classes(
        slope_classes=slope_classes,
        transform=transform,
        output_file=(OUTPUT_DIR / "slope_angle_classes_regions.png"),
        geometries=geometries,
        label_points=label_points,
        axis_off=True,
    )

    plot_slope_histogram(
        valid_slopes=slope[valid_mask & np.isfinite(slope)],
        output_file=(OUTPUT_DIR / "slope_angle_histogram.png"),
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


def calculate_monthly_climatology(rainfall):
    """
    Mean rainfall total for each calendar month (mm/month), averaged over years.

    Sum daily rainfall within every month, then average each calendar month
    across all years. Result dims: (month, N, E).
    """
    rainfall = rainfall.where(np.isfinite(rainfall))
    monthly_totals = rainfall.resample(time="MS").sum(
        dim="time",
        skipna=True,
        min_count=1,
    )
    climatology = monthly_totals.groupby("time.month").mean(
        dim="time",
        skipna=True,
    )
    climatology.name = "mean_monthly_rainfall"
    return climatology


def monthly_climatology_to_rasters(
    monthly_climatology,
) -> tuple[list, np.ndarray, rasterio.Affine]:
    """Convert the (month, N, E) climatology into a north-up [12, H, W] stack."""
    months = [int(m) for m in monthly_climatology["month"].values]
    arrays = []
    transform = None
    for month in months:
        array, transform = rainfall_to_raster(monthly_climatology.sel(month=month))
        arrays.append(np.asarray(array, dtype=np.float32))
    return months, np.stack(arrays, axis=0), transform


def rainfall_by_region_monthly(
    monthly_stack: np.ndarray,
    months: list,
    region_raster: np.ndarray,
    region_ids: list,
) -> pd.DataFrame:
    """Mean monthly rainfall (mm/month) per region -> long DataFrame."""
    rows = []
    for index, month in enumerate(months):
        grid = monthly_stack[index]
        finite = np.isfinite(grid)
        for region_id in region_ids:
            values = grid[finite & (region_raster == region_id)]
            rows.append(
                {
                    "region_id": region_id,
                    "month": month,
                    "mean_monthly_rainfall_mm": (
                        float(np.mean(values)) if values.size else np.nan
                    ),
                    "cell_count": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def rainfall_by_slope_class_monthly(
    monthly_stack: np.ndarray,
    months: list,
    slope_classes: np.ndarray,
) -> pd.DataFrame:
    """Monthly rainfall statistics per slope class -> long DataFrame."""
    rows = []
    for index, month in enumerate(months):
        grid = monthly_stack[index]
        for class_id, label in enumerate(SLOPE_LABELS, start=1):
            values = grid[(slope_classes == class_id) & np.isfinite(grid)]
            if values.size == 0:
                mean_v = median_v = std_v = min_v = max_v = np.nan
            else:
                mean_v = float(np.mean(values))
                median_v = float(np.median(values))
                std_v = float(np.std(values))
                min_v = float(np.min(values))
                max_v = float(np.max(values))
            rows.append(
                {
                    "class_id": class_id,
                    "slope_class": label,
                    "month": month,
                    "cell_count": int(values.size),
                    "mean_monthly_rainfall_mm": mean_v,
                    "median_monthly_rainfall_mm": median_v,
                    "std_monthly_rainfall_mm": std_v,
                    "min_monthly_rainfall_mm": min_v,
                    "max_monthly_rainfall_mm": max_v,
                }
            )

    plt.figure(figsize=(12, 6))
    for class_id, label in enumerate(SLOPE_LABELS, start=1):
        class_data = pd.DataFrame(rows)[pd.DataFrame(rows)["class_id"] == class_id]
        plt.plot(
            class_data["month"],
            class_data["mean_monthly_rainfall_mm"],
            marker="o",
            label=label,
        )
    plt.xlabel("Month")
    plt.ylabel("Mean monthly rainfall (mm/month)")
    plt.title("Mean monthly rainfall by slope class")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.legend(title="Slope class", loc="upper right", fontsize="small", frameon=False)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mean_monthly_rainfall_by_slope_class.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return pd.DataFrame(rows)


def plot_region_monthly_rainfall(
    region_monthly: pd.DataFrame,
    monthly_file: Path,
    cumulative_file: Path,
) -> None:
    """
    Two line plots, one line per region: mean monthly rainfall, and the
    cumulative total accumulating through the year.
    """
    pivot = region_monthly.pivot(
        index="month",
        columns="region_id",
        values="mean_monthly_rainfall_mm",
    ).sort_index()

    markers = ["o", "s", "^", "D"]

    for data, ylabel, title, out_file, legend_loc in [
        (
            pivot,
            "Mean monthly rainfall (mm/month)",
            "Mean monthly rainfall by region",
            monthly_file,
            "upper right",
        ),
        (
            pivot.cumsum(axis=0),
            "Cumulative rainfall (mm)",
            "Cumulative rainfall through the year by region",
            cumulative_file,
            "upper left",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, region_id in enumerate(data.columns):
            marker_custom = markers[(i // 10) % len(markers)]
            ax.plot(
                data.index,
                data[region_id],
                marker=marker_custom,
                linewidth=1,
                markersize=3,
                label=str(region_id),
            )
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
        ax.legend(
            title="Region",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),  # outside the axes on the right
            ncol=2,  # horizontal layout
            fontsize="small",
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
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

        # Monthly climatology (mm/month per calendar month) as a [12, H, W] stack.
        monthly_climatology = calculate_monthly_climatology(rainfall)
        months, monthly_stack, _ = monthly_climatology_to_rasters(monthly_climatology)

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

    # --- Monthly rainfall differentiation -----------------------------------

    # (a) Monthly rainfall per slope class (parallels the annual table above).
    monthly_by_class = rainfall_by_slope_class_monthly(
        monthly_stack=monthly_stack,
        months=months,
        slope_classes=slope_on_rainfall_grid,
    )
    monthly_by_class.to_csv(
        OUTPUT_DIR / "rainfall_by_slope_class_monthly.csv",
        index=False,
        float_format="%.3f",
    )

    # (b) Monthly rainfall per region: CSV, animation, and accumulation plots.
    geometries, label_points, region_ids = data_loader.load_regions()

    region_raster = data_loader.rasterize_drought_regions(
        rainfall_shape=monthly_stack.shape[1:],
        rainfall_transform=rainfall_transform,
        required_region_ids=region_ids,
    )

    region_monthly = rainfall_by_region_monthly(
        monthly_stack=monthly_stack,
        months=months,
        region_raster=region_raster,
        region_ids=region_ids,
    )
    region_monthly.to_csv(
        OUTPUT_DIR / "rainfall_by_region_monthly.csv",
        index=False,
        float_format="%.3f",
    )

    region_map.animate_monthly_regions(
        region_monthly,
        geometries,
        region_col="region_id",
        month_col="month",
        value_col="mean_monthly_rainfall_mm",
        label_points=label_points,
        output_file=(OUTPUT_DIR / "rainfall_by_region_monthly.gif"),
        fps=2,
        cmap="Blues",
        title="Mean monthly rainfall",
        cbar_label="Rainfall (mm/month)",
    )

    plot_region_monthly_rainfall(
        region_monthly=region_monthly,
        monthly_file=(OUTPUT_DIR / "rainfall_by_region_monthly.png"),
        cumulative_file=(OUTPUT_DIR / "rainfall_by_region_cumulative.png"),
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
