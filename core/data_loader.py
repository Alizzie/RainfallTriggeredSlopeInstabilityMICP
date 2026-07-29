"""
data_loader.py — Centralized data ingestion module.
Handles all NetCDF, CSV, raster, inventory and region-geometry loading to keep
execution scripts clean.
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr

# --- Standardized File Paths ---

# Legacy per-segment boundary CSVs (kept only for backward compatibility; the
# region geometry now comes from the GeoJSON below).
PATH_COORD = "data/trockenheit_grenzcoord/data_region{}.csv"

# Official BAFU drought-region polygons (all multipart segments), EPSG:2056.
# Feature id == calibration region_id (31-68), so no offset mapping is needed.
PATH_REGION_GEOJSON = "data/trockenheitsindex_2_2056.json"

PATH_RAIN = (
    "data/rhiresD/"
    "ogd-surface-derived-grid-archive."
    "rhiresd_ch01h.swiss.lv95_{}0101000000_{}1231000000.nc"
)

PATH_BAFU = "data/soil_moisture_history/weekly_historic_regions.csv"
PATH_CALIB = "output/02_calibration/calibration_results.csv"
PATH_INVENTORY = "data/wsl_inventory/wsl_landslide.csv"
PATH_WSL_USABLE = "data/wsl_inventory/wsl_usable_events.csv"


# =====================================================================
# Region geometry (GeoJSON) — single source of truth for drought regions
# =====================================================================

_REGION_GEOM_CACHE = None  # {region_id: geojson geometry dict}
_REGION_PARTS_CACHE = None  # {region_id: [shapely Polygon parts]}


def _feature_region_id(feature: dict) -> int:
    """Region ID of a GeoJSON feature (top-level id, else properties.ID)."""
    if feature.get("id") is not None:
        return int(feature["id"])
    props = feature.get("properties") or {}
    if props.get("ID") is not None:
        return int(props["ID"])
    raise ValueError("Region feature has neither 'id' nor properties.ID.")


def load_region_geometries(required_region_ids=None) -> dict:
    """
    Return {region_id: geojson geometry dict}, cached.

    If required_region_ids is given, verify all are present and return just
    those. No shapely dependency (raw geojson, suitable for rasterize).
    """
    global _REGION_GEOM_CACHE

    if _REGION_GEOM_CACHE is None:
        if not os.path.exists(PATH_REGION_GEOJSON):
            raise FileNotFoundError(f"Region GeoJSON missing: {PATH_REGION_GEOJSON}")
        with open(PATH_REGION_GEOJSON) as handle:
            data = json.load(handle)
        _REGION_GEOM_CACHE = {
            _feature_region_id(f): f["geometry"] for f in data["features"]
        }

    if required_region_ids is None:
        return _REGION_GEOM_CACHE

    missing = set(required_region_ids) - set(_REGION_GEOM_CACHE)
    if missing:
        raise FileNotFoundError(
            "Region GeoJSON is missing calibration region IDs: "
            + ", ".join(str(r) for r in sorted(missing))
        )
    return {rid: _REGION_GEOM_CACHE[rid] for rid in required_region_ids}


def rasterize_drought_regions(
    rainfall_shape,
    rainfall_transform,
    required_region_ids,
    all_touched=False,
):
    """
    Rasterise drought-region polygons (all parts) onto a grid.

    Values are calibration region IDs. Prints a presence/missing diagnostic.
    """
    from rasterio.features import rasterize

    geometries = load_region_geometries(required_region_ids)
    shapes = [(geometries[rid], int(rid)) for rid in sorted(required_region_ids)]

    region_ids = rasterize(
        shapes=shapes,
        out_shape=rainfall_shape,
        transform=rainfall_transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.int32,
    )

    present = {int(v) for v in np.unique(region_ids)} - {0}
    missing = set(required_region_ids) - present
    print(f"LOG: regions rasterised {len(present)}/{len(required_region_ids)}.")
    if missing:
        print(
            f"LOG: MISSING region IDs on the raster: {sorted(missing)} "
            "(a small part missed every cell centre; try all_touched=True)."
        )

    return region_ids


def _region_parts() -> dict:
    """Return {region_id: [shapely Polygon parts]} for ALL segments, cached."""
    global _REGION_PARTS_CACHE

    if _REGION_PARTS_CACHE is None:
        from shapely.geometry import shape

        if not os.path.exists(PATH_REGION_GEOJSON):
            raise FileNotFoundError(f"Region GeoJSON missing: {PATH_REGION_GEOJSON}")
        with open(PATH_REGION_GEOJSON) as handle:
            data = json.load(handle)

        parts = {}
        for feature in data["features"]:
            region_id = _feature_region_id(feature)
            geom = shape(feature["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            parts[region_id] = [p for p in polys if (not p.is_empty) and p.area > 0.0]
        _REGION_PARTS_CACHE = parts

    return _REGION_PARTS_CACHE


def region_representative_point(region_id):
    """
    (E, N) guaranteed inside the LARGEST part of the region.

    Replaces the old mean-of-vertices, which for a multipart region can land
    between segments — outside the region entirely.
    """
    parts = _region_parts().get(int(region_id))
    if not parts:
        raise KeyError(f"Region {region_id} not in {PATH_REGION_GEOJSON}.")
    largest = max(parts, key=lambda p: p.area)
    point = largest.representative_point()
    return float(point.x), float(point.y)


def assign_region(x, y, max_snap_m=None):
    """
    Region ID containing point (x, y) in LV95, or None.

    Exact point-in-polygon over ALL segments (multipart-native), not
    nearest-centroid. If the point is in no region and max_snap_m is set, snap
    to the nearest region whose boundary is within max_snap_m; else None. A
    *bounded* snap recovers points just outside a boundary without extrapolating
    region parameters across the country.
    """
    from shapely.geometry import Point

    point = Point(float(x), float(y))
    parts = _region_parts()

    for region_id, polys in parts.items():
        for poly in polys:
            if poly.contains(point):
                return int(region_id)

    if max_snap_m is None:
        return None

    best_id, best_dist = None, float("inf")
    for region_id, polys in parts.items():
        for poly in polys:
            distance = poly.distance(point)
            if distance < best_dist:
                best_id, best_dist = int(region_id), distance

    return best_id if best_dist <= float(max_snap_m) else None


def get_region_coordinates(region_id):
    """
    Representative LV95 (Easting, Northing) inside the region's largest part.

    Multipart-safe replacement for the old mean over a single-segment CSV.
    """
    return region_representative_point(region_id)


def get_region_params(x, y, calib, max_snap_m=2000.0):
    """
    (region_id, drainage, et) for the region CONTAINING (x, y).

    Point-in-polygon, not nearest-centroid: exact and multipart-safe. Returns
    (None, None, None) if the point lies in no region (and outside max_snap_m).
    """
    region_id = assign_region(x, y, max_snap_m=max_snap_m)
    if region_id is None:
        return None, None, None
    row = calib[calib["region_id"] == region_id]
    if row.empty:
        return region_id, None, None
    return (
        region_id,
        float(row["drainage"].iloc[0]),
        float(row["et"].iloc[0]),
    )


# =====================================================================
# Rainfall
# =====================================================================


def normalize_years(years):
    """Convert a single year or iterable of years into a list."""
    if isinstance(years, int):
        return [years]
    return list(years)


def rainfall_file_paths(years):
    """Return existing rainfall files for the requested years (missing skipped)."""
    paths = []
    for year in normalize_years(years):
        file_path = PATH_RAIN.format(year, year)
        if os.path.exists(file_path):
            paths.append(file_path)
        else:
            print(f"Warning: rainfall file missing for {year}: {file_path}")
    return paths


def load_rainfall(easting, northing, years):
    """Load and concatenate daily rainfall for one LV95 coordinate."""
    frames = []
    for file_path in rainfall_file_paths(years):
        with xr.open_dataset(file_path) as dataset:
            rainfall = dataset["RhiresD"].sel(E=easting, N=northing, method="nearest")
            frames.append(
                pd.Series(
                    np.nan_to_num(rainfall.values, nan=0.0),
                    index=pd.to_datetime(rainfall.time.values),
                )
            )
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def load_rainfall_grid(years, variable="RhiresD", chunks=None):
    """
    Load complete daily RhiresD grids for several years.

    Retains the spatial dimensions E and N (for map generation).
    """
    file_paths = rainfall_file_paths(years)
    if not file_paths:
        return None

    datasets = []
    try:
        for file_path in file_paths:
            dataset = xr.open_dataset(file_path, chunks=chunks)
            if variable not in dataset.data_vars:
                dataset.close()
                raise KeyError(
                    f"Variable '{variable}' not found in {file_path}. "
                    f"Available: {list(dataset.data_vars)}"
                )
            datasets.append(dataset[[variable]])

        combined = xr.concat(datasets, dim="time").sortby("time")
        rainfall = combined[variable]

        _, unique_indices = np.unique(rainfall["time"].values, return_index=True)
        rainfall = rainfall.isel(time=np.sort(unique_indices))
        return rainfall

    except Exception:
        for dataset in datasets:
            dataset.close()
        raise


# =====================================================================
# BAFU moisture, calibration, inventory
# =====================================================================


def load_bafu_moisture(region_id, year=None, interpolate_daily=False):
    """Load BAFU nFK data, optionally filtered by year and daily-interpolated."""
    df = pd.read_csv(
        PATH_BAFU,
        sep=",",
        skiprows=3,
        parse_dates=["measured_at"],
        dayfirst=True,
    )
    df = df[df["drought_region_id"] == region_id].copy()
    if year:
        df = df[df["measured_at"].dt.year == year]
    df = df.set_index("measured_at")
    nfk = df["soil_moisture_ufc"] / 100.0
    if interpolate_daily and not nfk.empty:
        nfk = nfk.resample("D").interpolate(method="linear")
    return nfk


def load_calibration_params(region_id):
    """Fetch optimized drainage and ET rates for a region."""
    if not os.path.exists(PATH_CALIB):
        return None, None
    calibration = pd.read_csv(PATH_CALIB)
    row = calibration[calibration["region_id"] == region_id]
    if row.empty:
        return None, None
    return float(row["drainage"].iloc[0]), float(row["et"].iloc[0])


def load_wsl_inventory():
    """Load the WSL landslide inventory and standardize coordinates and dates."""
    df = pd.read_csv(PATH_INVENTORY, skiprows=3)
    df = df.rename(
        columns={
            "x-coordinate": "x",
            "y-coordinate": "y",
            "date": "date",
            "name of municipality": "municipality",
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna(subset=["date", "x", "y"]).reset_index(drop=True)


def load_wsl_usable_inventory():
    """Load the filtered WSL inventory containing usable events."""
    if not os.path.exists(PATH_WSL_USABLE):
        raise FileNotFoundError(f"Usable inventory file missing: {PATH_WSL_USABLE}")
    df = pd.read_csv(PATH_WSL_USABLE)
    df = df.rename(
        columns={
            "x-coordinate": "x",
            "y-coordinate": "y",
            "date": "date",
            "name of municipality": "municipality",
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    print(f"Loaded {len(df)} usable events from the WSL inventory.")
    return df.dropna(subset=["date", "x", "y"]).reset_index(drop=True)
