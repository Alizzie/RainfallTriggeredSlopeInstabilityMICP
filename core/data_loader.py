"""
data_loader.py — Centralized data ingestion module.
Handles all NetCDF, CSV, raster, inventory and region-geometry loading to keep
execution scripts clean.
"""

import csv
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import shape
from shapely.geometry import Point
from rasterio.features import rasterize

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
PATH_CALIB = "output/temporal/02_calibration/calibration_results.csv"
PATH_INVENTORY = "data/wsl_inventory/wsl_landslide.csv"
PATH_WSL_USABLE = "data/wsl_inventory/wsl_usable_events.csv"
STORME_PATH = "data/wsl_inventory/hangmuren_storme.csv"


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


def load_regions(region_ids=None):
    """
    Return (geometries, label_points, ids) for all drought regions.

    label_points are guaranteed-inside anchors (largest part) used to write the
    region ID; ids is the sorted list of region IDs.
    """
    geometries = load_region_geometries()

    if region_ids is None:
        ids = sorted(geometries)
    else:
        ids = [rid for rid in region_ids if rid in geometries]
        geometries = {rid: geometries[rid] for rid in ids}

    label_points = {rid: region_representative_point(rid) for rid in ids}
    return geometries, label_points, ids


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
        return None, None, None
    calibration = pd.read_csv(PATH_CALIB)
    row = calibration[calibration["region_id"] == region_id]
    if row.empty:
        return None, None, None

    amplitude = (
        float(row["et_amplitude"].iloc[0]) if "et_amplitude" in row.columns else 0.0
    )
    return float(row["drainage"].iloc[0]), float(row["et"].iloc[0]), amplitude


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
    df["x"], df["y"] = zip(*df.apply(lambda row: to_lv95(row["x"], row["y"]), axis=1))
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
    df["date"] = pd.to_datetime(
        df["date"], format="mixed", errors="coerce", dayfirst=True
    )
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["x"], df["y"] = zip(*df.apply(lambda row: to_lv95(row["x"], row["y"]), axis=1))
    print(f"Loaded {len(df)} usable events from the WSL inventory.")
    return df.dropna(subset=["date", "x", "y"]).reset_index(drop=True)


def load_storme_inventory():
    """Load the StormE landslide inventory and standardize coordinates and dates."""
    if not os.path.exists(STORME_PATH):
        raise FileNotFoundError(f"StormE inventory file missing: {STORME_PATH}")
    df = pd.read_csv(
        STORME_PATH,
        sep=",",
        skiprows=2,
        engine="python",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8-sig",
        on_bad_lines="warn",
    )

    df = df.rename(
        columns={
            "X-Koordinate": "x",
            "Y-Koordinate": "y",
            "Datum": "date",
            "Neigung": "slope",
        }
    )
    df["date"] = pd.to_datetime(
        df["date"], format="mixed", errors="coerce", dayfirst=True
    )
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["slope"] = pd.to_numeric(df["slope"], errors="coerce")
    print(f"Loaded {len(df)} events from the StormE inventory.")
    return df.dropna(subset=["date", "x", "y"]).reset_index(drop=True)


def flag_likely_duplicates(combined_df, date_tol_days=2, dist_tol_m=500):
    """
    Mark StormE events that likely duplicate a WSL event (same time, same
    place), so the 5 regions covered by both sources don't get double-counted.
    Adds a boolean 'likely_duplicate' column; only StormE rows can be flagged,
    since StormE is the more precise source and is kept in a tie.
    """
    combined_df["likely_duplicate"] = False
    wsl = combined_df[combined_df["source"] == "wsl"]

    for i in combined_df.index[combined_df["source"] == "storme"]:
        row = combined_df.loc[i]
        close_in_time = wsl[
            (wsl["date"] - row["date"]).abs() <= pd.Timedelta(days=date_tol_days)
        ]
        if close_in_time.empty:
            continue
        dist = np.hypot(close_in_time["x"] - row["x"], close_in_time["y"] - row["y"])
        if (dist <= dist_tol_m).any():
            combined_df.loc[i, "likely_duplicate"] = True

    n_dupes = combined_df["likely_duplicate"].sum()
    print(f"Flagged {n_dupes} StormE events as likely duplicates of a WSL event.")
    return combined_df


def load_combined_inventory(drop_duplicates=True):
    """
    Load both WSL and StormE inventories, tag each row's source, flag likely
    duplicates in the overlapping post-2005 regions, and combine.

    drop_duplicates=True excludes flagged StormE duplicates from the combined
    frame (keeping the WSL copy); pass False to keep everything with the flag
    intact for inspection instead.
    """
    wsl_df = load_wsl_usable_inventory()
    storme_df = load_storme_inventory()

    wsl_df["source"] = "wsl"
    storme_df["source"] = "storme"

    combined_df = pd.concat([wsl_df, storme_df], ignore_index=True)
    combined_df = flag_likely_duplicates(combined_df)

    if drop_duplicates:
        combined_df = combined_df[~combined_df["likely_duplicate"]].reset_index(
            drop=True
        )

    print(
        f"Combined inventory contains {len(combined_df)} events "
        f"({(combined_df['source'] == 'wsl').sum()} WSL, "
        f"{(combined_df['source'] == 'storme').sum()} StormE)."
    )
    return combined_df


def to_lv95(a, b):
    """
    Convert a raw (a, b) coordinate pair to LV95 (easting, northing).
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan, np.nan

    a, b = float(a), float(b)

    # LV95 already in correct E, N order
    if 2_400_000 <= a <= 2_900_000 and 1_000_000 <= b <= 1_400_000:
        return a, b

    # LV95 supplied as N, E
    if 1_000_000 <= a <= 1_400_000 and 2_400_000 <= b <= 2_900_000:
        return b, a

    # LV03 already in correct E, N order
    if 400_000 <= a <= 900_000 and 0 <= b <= 400_000:
        return a + 2_000_000, b + 1_000_000

    # LV03 supplied as N, E
    if 0 <= a <= 400_000 and 400_000 <= b <= 900_000:
        return b + 2_000_000, a + 1_000_000

    # Unknown or invalid coordinate convention
    return np.nan, np.nan
