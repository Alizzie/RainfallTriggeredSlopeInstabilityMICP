"""
Yearly rainfall-driven soil-moisture / slope-stability climatology (table only).

1. What
    Runs the calibrated bucket model over historical daily rainfall and reports,
    per year, three national summary numbers: mean saturation, the share of
    steep soil that is unstable on the year's wettest day, and the mean number
    of days below FoS = 1.

2. Input
    - Daily RhiresD rainfall grids (core.data_loader.load_rainfall_grid).
    - Per-region calibrated drainage/ET (core.data_loader.PATH_CALIB).
    - Drought-region polygons (region_geometry, all multipart segments).
    - Slope raster (degrees).

3. Output
    - yearly_climatology.csv with columns:
        year, mean_saturation, design_m_median, unstable_pct,
        mean_days_fos_below_one
    - design_saturation_m_25m.tif: one design saturation field aggregated over
      all years, on the 25 m map grid, to replace m_pp = 1.0 in spatial_fos.py.

4. Workflow
    Bucket runs on the coarse rainfall grid (hydrology is coarse by nature).
    Per year we keep the mean and max pore-pressure ratio and a coarse
    day-below-1 count. The unstable share is evaluated at year end on a fine
    slope grid (15-45 deg mask) from the up-sampled yearly-max m, exploiting
    FoS being linear in m so max m gives the worst-day FoS exactly.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import constants as const
from core import data_loader as dl
from core import physics

# --- Configuration ---

OUTPUT_DIR = Path("output/statistics/yearly_climatology")
SLOPE_RASTER = Path("data/swissalti_slope/slope_deg_25m_ch.tif")
ANALYSIS_YEARS = range(1991, 2026)
OUTPUT_CRS = "EPSG:2056"

FOS_MAP_RESOLUTION = 100.0

# Which saturation drives the "unstable share".
#   None -> the annual MAX of m per cell (worst single day; saturates near 1).
#   e.g. 99.0 -> the 99th percentile of daily m per cell (a design saturation
#        that stays below the ceiling and discriminates years -> feeds the
#        rainfall-derived design saturation that replaces m_pp = 1.0).
DESIGN_M_PERCENTILE = 99.0

# --- Design-saturation raster export (feeds spatial_fos.py, replaces m_pp=1.0) ---
# Write a single design-saturation field aggregated over ALL years, on the grid
# of MAP_REFERENCE_RASTER (your 25 m map) so spatial_fos.py can load it directly.
WRITE_DESIGN_RASTER = True
MAP_REFERENCE_RASTER = SLOPE_RASTER  # align the output to this grid (25 m)
DESIGN_RASTER_PATH = OUTPUT_DIR / "design_saturation_m_25m.tif"

# How to collapse the 35 yearly design_m fields into one, per cell.
#   "median" (default) -> the design saturation of a typical year (robust).
#   "mean" | "max" | "p90" | "p95" also available.
MULTIYEAR_AGG = "median"

# Spatial granularity of the exported field:
#   "region" -> ONE design value per drought region, rasterised at 25 m
#               (clean region boundaries; matches the per-region calibration;
#               no 2 km checkerboard). Recommended.
#   "cell"   -> the coarse per-cell field resampled (nearest) to 25 m
#               (keeps within-region rainfall variation, but 2 km-blocky).
DESIGN_M_SPATIAL = "region"  # /"cell"

TIME_NAME, EASTING_NAME, NORTHING_NAME = "time", "E", "N"
RAINFALL_CHUNKS = {TIME_NAME: 31}
UNSTABLE_FOS_THRESHOLD = 1.0

# Infinite-slope validity mask (degrees). Without it, near-flat cells send the
# driving shear tau = gamma*H*sin(b)*cos(b) -> 0 and FoS diverges, corrupting
# any FoS average. Add BETA_MIN/BETA_MAX to constants.py (15 / 45).
BETA_MIN = const.BETA_MIN
BETA_MAX = const.BETA_MAX

INITIAL_SATURATION = const.M0
COHESION = const.C  # kPa; note: 0 in constants (cohesionless)
SOIL_UNIT_WEIGHT = const.GAMMA
WATER_UNIT_WEIGHT = const.GAMMA_W
SOIL_DEPTH_VERTICAL = const.H_V  # vertical failure depth -> FoS
FRICTION_ANGLE_DEG = const.PHI_DEG
POROSITY = const.N
SOIL_THICKNESS_PERP = const.H_PERP  # perpendicular thickness -> bucket capacity
PORE_PRESSURE_ONSET = const.S_PP_ONSET_DEFAULT


# --- Rainfall ---


def load_rainfall():
    rainfall = dl.load_rainfall_grid(years=ANALYSIS_YEARS, chunks=RAINFALL_CHUNKS)
    if rainfall is None:
        raise FileNotFoundError("No rainfall files for the configured years.")
    if isinstance(rainfall, __import__("xarray").Dataset):
        variables = list(rainfall.data_vars)
        if len(variables) != 1:
            raise ValueError(f"Ambiguous rainfall variables: {variables}")
        rainfall = rainfall[variables[0]]
    return rainfall


def rainfall_grid_geometry(rainfall):
    eastings = np.asarray(rainfall[EASTING_NAME].values, dtype=float)
    northings = np.asarray(rainfall[NORTHING_NAME].values, dtype=float)
    x_res = abs(float(np.median(np.diff(eastings))))
    y_res = abs(float(np.median(np.diff(northings))))
    transform = from_origin(
        min(eastings) - x_res / 2, max(northings) + y_res / 2, x_res, y_res
    )
    shape = (rainfall.sizes[NORTHING_NAME], rainfall.sizes[EASTING_NAME])
    print(
        f"LOG: rainfall grid {x_res:.0f} m; share evaluated at "
        f"{FOS_MAP_RESOLUTION:.0f} m."
    )
    return transform, shape


def orient_rainfall_slice(rainfall_slice):
    rainfall_slice = rainfall_slice.transpose(NORTHING_NAME, EASTING_NAME)
    eastings = np.asarray(rainfall_slice[EASTING_NAME].values, dtype=float)
    northings = np.asarray(rainfall_slice[NORTHING_NAME].values, dtype=float)
    values = np.asarray(rainfall_slice.values, dtype=np.float32)
    if eastings[0] > eastings[-1]:
        values = np.fliplr(values)
    if northings[0] < northings[-1]:
        values = np.flipud(values)
    return values


# --- Grids ---


def reproject_raster_to_grid(raster_file, shape, transform, resampling):
    destination = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(raster_file) as source:
        reproject(
            source=source.read(1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=OUTPUT_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return np.where(np.isfinite(destination), destination, np.nan).astype(np.float32)


def build_fine_slope(rainfall_transform, rainfall_shape):
    """Fine slope grid + its 15-45 deg validity mask, over the rainfall extent."""
    left, top = rainfall_transform.c, rainfall_transform.f
    right = left + rainfall_transform.a * rainfall_shape[1]
    bottom = top + rainfall_transform.e * rainfall_shape[0]
    width = int(round((right - left) / FOS_MAP_RESOLUTION))
    height = int(round((top - bottom) / FOS_MAP_RESOLUTION))
    transform = from_origin(left, top, FOS_MAP_RESOLUTION, FOS_MAP_RESOLUTION)
    shape = (height, width)
    slope = reproject_raster_to_grid(
        SLOPE_RASTER, shape, transform, Resampling.bilinear
    )
    mask = np.isfinite(slope) & (slope >= BETA_MIN) & (slope <= BETA_MAX)
    print(
        f"LOG: fine slope {shape[1]}x{shape[0]}; "
        f"{int(mask.sum())} pixels in {BETA_MIN:.0f}-{BETA_MAX:.0f} deg."
    )
    return slope, transform, shape, mask


def reproject_array_to_grid(values, src_transform, dst_shape, dst_transform):
    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(values, dtype=np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=OUTPUT_CRS,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=OUTPUT_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,  # honest: model has no sub-cell m gradient
    )
    return destination


def load_calibration_table():
    calibration = pd.read_csv(dl.PATH_CALIB)
    for column in ("region_id", "drainage", "et"):
        if column not in calibration.columns:
            raise ValueError(f"Calibration table missing '{column}'.")
    calibration = calibration.copy()
    calibration["region_id"] = calibration["region_id"].astype(int)
    return calibration


def calibrated_parameter_maps(region_ids, calibration):
    drainage = np.full(region_ids.shape, np.nan, dtype=np.float32)
    et = np.full(region_ids.shape, np.nan, dtype=np.float32)
    for row in calibration.itertuples(index=False):
        mask = region_ids == int(row.region_id)
        drainage[mask] = float(row.drainage)
        et[mask] = float(row.et)
    return drainage, et


# --- Bucket + FoS ---


def update_bucket(water_mm, rainfall_mm, drainage_rate, et_rate, valid_mask, cap_mm):
    onset_mm = PORE_PRESSURE_ONSET * cap_mm
    rainfall_mm = np.where(np.isfinite(rainfall_mm), np.maximum(rainfall_mm, 0.0), 0.0)
    excess = np.maximum(water_mm - onset_mm, 0.0)
    updated = water_mm + rainfall_mm - drainage_rate * excess - et_rate
    updated = np.clip(updated, 0.0, cap_mm)
    updated[~valid_mask] = np.nan
    return updated.astype(np.float32)


def coarse_days_below_one_increment(slope_deg, pore_pressure, valid_mask):
    """Daily FoS on the coarse grid, returns boolean FoS<1 (flat cells never fire)."""
    beta = np.deg2rad(slope_deg)
    phi = np.deg2rad(FRICTION_ANGLE_DEG)
    fos = np.full(slope_deg.shape, np.nan, dtype=np.float32)
    calc = valid_mask & np.isfinite(beta) & (beta > 0.0) & (beta < np.pi / 2)
    fos[calc] = np.asarray(
        physics.compute_fos(
            c=COHESION,
            gamma=SOIL_UNIT_WEIGHT,
            gamma_w=WATER_UNIT_WEIGHT,
            h_v=SOIL_DEPTH_VERTICAL,
            beta_rad=beta[calc],
            phi_rad=phi,
            m_array=pore_pressure[calc],
        ),
        dtype=np.float32,
    )
    return np.isfinite(fos) & (fos < UNSTABLE_FOS_THRESHOLD)


def unstable_pct_at_m(fine_slope, fine_mask, m_fine):
    """Percent of 15-45 deg soil pixels with FoS < 1 at the given m field."""
    beta = np.deg2rad(fine_slope)
    phi = np.deg2rad(FRICTION_ANGLE_DEG)
    evaluate = fine_mask & np.isfinite(m_fine)
    if not np.any(evaluate):
        return np.nan
    fos = physics.compute_fos(
        c=COHESION,
        gamma=SOIL_UNIT_WEIGHT,
        gamma_w=WATER_UNIT_WEIGHT,
        h_v=SOIL_DEPTH_VERTICAL,
        beta_rad=beta[evaluate],
        phi_rad=phi,
        m_array=m_fine[evaluate],
    )
    fos = np.asarray(fos, dtype=np.float32)
    return float(100.0 * np.mean(fos < UNSTABLE_FOS_THRESHOLD))


# --- Design-saturation raster export ---


def multiyear_aggregate(design_m_years):
    """Collapse the per-year coarse design_m grids into one field, per cell."""
    stack = np.stack(design_m_years, axis=0)  # (years, H, W), NaN outside
    if MULTIYEAR_AGG == "median":
        agg = np.nanmedian(stack, axis=0)
    elif MULTIYEAR_AGG == "mean":
        agg = np.nanmean(stack, axis=0)
    elif MULTIYEAR_AGG == "max":
        agg = np.nanmax(stack, axis=0)
    elif MULTIYEAR_AGG.startswith("p"):
        agg = np.nanpercentile(stack, float(MULTIYEAR_AGG[1:]), axis=0)
    else:
        raise ValueError(f"Unknown MULTIYEAR_AGG: {MULTIYEAR_AGG}")
    return agg.astype(np.float32)


def _reference_grid():
    """(transform, (height, width), crs) of the 25 m map reference raster."""
    with rasterio.open(MAP_REFERENCE_RASTER) as source:
        return source.transform, (source.height, source.width), source.crs


def _write_geotiff(values, transform, crs, path):
    nodata = -9999.0
    out = np.where(np.isfinite(values), values, nodata).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": out.shape[0],
        "width": out.shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    print(f"Design saturation written to {path} ({out.shape[1]}x{out.shape[0]}).")


def write_design_saturation(
    agg_coarse, rainfall_transform, region_ids_coarse, required_region_ids
):
    """
    Write the aggregated design saturation on the 25 m reference grid.

    "region": one value per drought region (median of that region's coarse
    design_m over all cells), rasterised at 25 m -> clean region boundaries.
    "cell":   the coarse per-cell field resampled (nearest) to 25 m.
    """
    transform, shape, crs = _reference_grid()

    if DESIGN_M_SPATIAL == "cell":
        fine = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=np.ascontiguousarray(agg_coarse, dtype=np.float32),
            destination=fine,
            src_transform=rainfall_transform,
            src_crs=OUTPUT_CRS,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,  # coarse hydrology: no fake gradients
        )
        _write_geotiff(fine, transform, crs, DESIGN_RASTER_PATH)
        return

    if DESIGN_M_SPATIAL != "region":
        raise ValueError(f"Unknown DESIGN_M_SPATIAL: {DESIGN_M_SPATIAL}")

    # one design value per region
    region_value = {}
    for rid in sorted(required_region_ids):
        cells = agg_coarse[(region_ids_coarse == rid) & np.isfinite(agg_coarse)]
        region_value[rid] = float(np.median(cells)) if cells.size else np.nan
        print(f"  region {rid}: design m = {region_value[rid]:.3f}")

    # rasterise regions on the 25 m grid, then map region id -> design value
    region_ids_fine = dl.rasterize_drought_regions(
        shape, transform, required_region_ids
    )
    fine = np.full(shape, np.nan, dtype=np.float32)
    for rid, value in region_value.items():
        if np.isfinite(value):
            fine[region_ids_fine == rid] = value
    _write_geotiff(fine, transform, crs, DESIGN_RASTER_PATH)


# --- Run ---


def run(
    rainfall,
    coarse_slope,
    drainage_rate,
    et_rate,
    valid_mask,
    rainfall_transform,
    fine_slope,
    fine_transform,
    fine_shape,
    fine_mask,
):
    cap_mm = POROSITY * SOIL_THICKNESS_PERP * 1000.0
    water_mm = np.full(coarse_slope.shape, np.nan, dtype=np.float32)
    water_mm[valid_mask] = INITIAL_SATURATION * cap_mm

    time_values = pd.DatetimeIndex(rainfall[TIME_NAME].values)
    rows = []
    design_m_years = []  # one coarse design_m grid per year (for the raster export)
    current_year, acc = None, None

    def new_acc():
        acc = {
            "sat_sum": np.zeros(coarse_slope.shape, np.float64),
            "sat_count": np.zeros(coarse_slope.shape, np.uint32),
            "days_below": np.zeros(coarse_slope.shape, np.uint32),
        }
        if DESIGN_M_PERCENTILE is None:
            acc["max_m"] = np.full(coarse_slope.shape, -np.inf, np.float32)
        else:
            # daily m for the valid cells only (keeps the per-year stack small)
            acc["m_daily"] = []
        return acc

    def design_m_coarse(a):
        """The saturation field that drives instability: annual max or percentile."""
        if DESIGN_M_PERCENTILE is None:
            return np.where(np.isfinite(a["max_m"]), a["max_m"], np.nan).astype(
                np.float32
            )
        # reconstruct (days, n_valid) -> percentile per valid cell -> full grid
        stack = np.stack(a["m_daily"], axis=0)  # (days, n_valid)
        pct = np.nanpercentile(stack, DESIGN_M_PERCENTILE, axis=0)
        grid = np.full(coarse_slope.shape, np.nan, np.float32)
        grid[valid_mask] = pct.astype(np.float32)
        return grid

    def close(year, a):
        sat = np.full(coarse_slope.shape, np.nan, np.float32)
        ok = a["sat_count"] > 0
        sat[ok] = (a["sat_sum"][ok] / a["sat_count"][ok]).astype(np.float32)

        design_m = design_m_coarse(a)
        design_m_years.append(design_m)
        design_m_fine = reproject_array_to_grid(
            design_m, rainfall_transform, fine_shape, fine_transform
        )
        rows.append(
            {
                "year": int(year),
                "mean_saturation": float(np.nanmean(sat)),
                "design_m_median": float(np.nanmedian(design_m[valid_mask])),
                "unstable_pct": unstable_pct_at_m(fine_slope, fine_mask, design_m_fine),
                # mean over VALID Swiss cells only (outside cells are 0, not NaN,
                # so np.nanmean over the full grid would halve this number).
                "mean_days_fos_below_one": float(a["days_below"][valid_mask].mean()),
            }
        )
        print(f"Finished {year}.")

    for i, ts in enumerate(time_values):
        year = int(ts.year)
        if current_year is None:
            current_year, acc = year, new_acc()
        if year != current_year:
            close(current_year, acc)
            current_year, acc = year, new_acc()

        rain = orient_rainfall_slice(rainfall.isel({TIME_NAME: i}))
        water_mm = update_bucket(
            water_mm, rain, drainage_rate, et_rate, valid_mask, cap_mm
        )
        saturation = (water_mm / cap_mm).astype(np.float32)
        m = physics.pore_pressure_ratio(saturation, PORE_PRESSURE_ONSET).astype(
            np.float32
        )

        sv = np.isfinite(saturation)
        acc["sat_sum"][sv] += saturation[sv]
        acc["sat_count"][sv] += 1
        if DESIGN_M_PERCENTILE is None:
            mv = np.isfinite(m)
            acc["max_m"][mv] = np.maximum(acc["max_m"][mv], m[mv])
        else:
            acc["m_daily"].append(m[valid_mask].astype(np.float32))
        acc["days_below"][
            coarse_days_below_one_increment(coarse_slope, m, valid_mask)
        ] += 1

    if current_year is not None:
        close(current_year, acc)

    return pd.DataFrame(rows), design_m_years


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rainfall = load_rainfall()
    try:
        rainfall_transform, rainfall_shape = rainfall_grid_geometry(rainfall)
        calibration = load_calibration_table()
        required = set(calibration["region_id"].astype(int))

        coarse_slope = reproject_raster_to_grid(
            SLOPE_RASTER, rainfall_shape, rainfall_transform, Resampling.bilinear
        )
        region_ids = dl.rasterize_drought_regions(
            rainfall_shape, rainfall_transform, required
        )
        fine_slope, fine_transform, fine_shape, fine_mask = build_fine_slope(
            rainfall_transform, rainfall_shape
        )
        drainage_rate, et_rate = calibrated_parameter_maps(region_ids, calibration)

        valid_mask = (
            np.isfinite(coarse_slope)
            & np.isfinite(drainage_rate)
            & np.isfinite(et_rate)
            & (region_ids > 0)
        )
        if not np.any(valid_mask):
            raise ValueError("No valid model cells.")

        level = (
            "annual max m"
            if DESIGN_M_PERCENTILE is None
            else f"p{DESIGN_M_PERCENTILE:g} of daily m"
        )
        print(
            f"Running {rainfall.sizes[TIME_NAME]} daily steps "
            f"(design saturation = {level})."
        )
        table, design_m_years = run(
            rainfall,
            coarse_slope,
            drainage_rate,
            et_rate,
            valid_mask,
            rainfall_transform,
            fine_slope,
            fine_transform,
            fine_shape,
            fine_mask,
        )
    finally:
        try:
            rainfall.close()
        except Exception:
            pass

    table.to_csv(
        OUTPUT_DIR / "yearly_climatology.csv", index=False, float_format="%.4f"
    )
    print("\n" + table.to_string(index=False))
    print("\nWritten to:", (OUTPUT_DIR / "yearly_climatology.csv").resolve())

    if WRITE_DESIGN_RASTER and design_m_years:
        print(
            f"\nAggregating {len(design_m_years)} years "
            f"({MULTIYEAR_AGG}, {DESIGN_M_SPATIAL}) into the design raster..."
        )
        agg_coarse = multiyear_aggregate(design_m_years)
        write_design_saturation(agg_coarse, rainfall_transform, region_ids, required)


if __name__ == "__main__":
    main()
