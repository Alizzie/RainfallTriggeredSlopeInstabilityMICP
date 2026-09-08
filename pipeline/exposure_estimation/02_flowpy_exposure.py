"""
02_flowpy_exposure.py - Intersect Flow-Py output with infrastructure and
conservation layers.
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

parser = argparse.ArgumentParser()
parser.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)

args = parser.parse_args()

FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
FLOWPY_INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"
OUTDIR = FLOWPY_DIR + "/exposure_summary"
os.makedirs(OUTDIR, exist_ok=True)

CELL_SIZE_M = 25.0
CELL_AREA_M2 = CELL_SIZE_M**2


AREA_LAYERS = {
    "forest": "data/swissTLM3D/tlm_forest_25m_ch.tif",
    "protection forest": "data/infra/schutzwald_rutschungen_25m.tif",
    "agricultural land": "data/infra/nutzungsgebiete.tif",
    "protected areas": "data/infra/protected_area_25m.tif",
}


LINE_LAYERS = {
    "roads": "data/infra/roads_binary_25m.tif",
    "railway": "data/infra/railway_binary_25m.tif",
}


BUILDINGS_ID_TIF = "data/infra/buildings_id_25m.tif"
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"


def read_aligned(path, ref_shape, label, optional=False):
    """Read a national raster, cropped to the Flow-Py window."""
    if not os.path.exists(path):
        if optional:
            print(f"[skip] {label}: {path} not found")
            return None
        raise FileNotFoundError(f"{label}: {path}")

    meta = json.load(open(f"{FLOWPY_INPUT_DIR}/meta.json"))
    r0, r1, c0, c1 = meta["crop_window"]

    with rasterio.open(path) as src:
        arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))

    if arr.shape != ref_shape:
        raise ValueError(f"{label} shape {arr.shape} != Flow-Py grid {ref_shape}")
    return arr


def main():
    # --- Flow-Py output = the affected area ---
    with rasterio.open(f"{FLOWPY_DIR}/z_delta.tif") as src:
        z_delta = src.read(1).astype(np.float32)
        shape = z_delta.shape

    with rasterio.open(f"{FLOWPY_DIR}/cell_counts.tif") as src:
        cell_counts = src.read(1).astype(np.float32)

    affected = np.isfinite(z_delta) & (z_delta > 0)
    print(f"grid: {shape}")
    print(
        f"affected cells: {int(affected.sum()):,} "
        f"({affected.sum() * CELL_AREA_M2 / 1e6:.1f} km2)"
    )

    rows = []

    # --- area layers ---
    for name, path in AREA_LAYERS.items():
        layer = read_aligned(path, shape, name, optional=True)
        if layer is None:
            continue
        mask = layer > 0
        hit = affected & mask
        total_ha = mask.sum() * CELL_AREA_M2 / 10_000
        hit_ha = hit.sum() * CELL_AREA_M2 / 10_000
        paths = cell_counts[hit]
        rows.append(
            {
                "layer": name,
                "unit": "ha",
                "total": round(total_ha, 1),
                "affected": round(hit_ha, 1),
                "pct_affected": (
                    round(100 * hit_ha / total_ha, 2) if total_ha else np.nan
                ),
                "median_paths": float(np.median(paths)) if paths.size else np.nan,
                "p90_paths": float(np.percentile(paths, 90)) if paths.size else np.nan,
                "max_paths": float(paths.max()) if paths.size else np.nan,
            }
        )

    # --- line layers ---
    for name, path in LINE_LAYERS.items():
        layer = read_aligned(path, shape, name, optional=True)
        if layer is None:
            continue
        mask = layer > 0
        hit = affected & mask
        total_m = mask.sum() * CELL_SIZE_M
        hit_m = hit.sum() * CELL_SIZE_M
        paths = cell_counts[hit]
        rows.append(
            {
                "layer": name,
                "unit": "m",
                "total": round(total_m, 0),
                "affected": round(hit_m, 0),
                "pct_affected": round(100 * hit_m / total_m, 2) if total_m else np.nan,
                "median_paths": (float(np.median(paths)) if paths.size else np.nan),
                "p90_paths": (
                    float(np.percentile(paths, 90)) if paths.size else np.nan
                ),
                "max_paths": (float(paths.max()) if paths.size else np.nan),
            }
        )

    # --- buildings ---
    buildings = read_aligned(BUILDINGS_ID_TIF, shape, "buildings", optional=True)
    if buildings is not None:
        total_ids = np.unique(buildings[buildings > 0])
        hit_cells = affected & (buildings > 0)
        hit_ids = np.unique(buildings[hit_cells])

        ids = buildings[hit_cells]
        paths_per_cell = cell_counts[hit_cells]
        per_building = (
            pd.Series(paths_per_cell).groupby(ids).max().to_numpy()
            if ids.size
            else np.array([])
        )
        rows.append(
            {
                "layer": "buildings",
                "unit": "count",
                "total": len(total_ids),
                "affected": len(hit_ids),
                "pct_affected": (
                    round(100 * len(hit_ids) / len(total_ids), 2)
                    if len(total_ids)
                    else np.nan
                ),
                "median_paths": (
                    float(np.median(per_building)) if per_building.size else np.nan
                ),
                "p90_paths": (
                    float(np.percentile(per_building, 90))
                    if per_building.size
                    else np.nan
                ),
                "max_paths": float(per_building.max()) if per_building.size else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/exposure_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n-> {OUTDIR}/exposure_summary.csv")


if __name__ == "__main__":
    main()
