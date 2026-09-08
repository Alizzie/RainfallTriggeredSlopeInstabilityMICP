"""
00_export_release_raster.py - Export a Flow-Py release raster + matching DEM.

Flow-Py treats every cell with a value > 0 as its own release cell with its own
path. There is no clustering step: the unstable mask IS the input.

USAGE
    python 00_export_release_raster.py                     # baseline
    python 00_export_release_raster.py --fos <path> --name micp
    python 00_export_release_raster.py --thin 1 --test-window 300
"""

import argparse
import json
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
from rasterio import Affine

# ============================================================== CONFIG =====
parser = argparse.ArgumentParser()

parser.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)
parser.add_argument(
    "--fos_map",
    default="m_pp_1.0/fos_bare.tif",
    help="FoS raster to use",
)
parser.add_argument(
    "--canton_ids",
    default=[15],
    type=lambda s: [int(x) for x in s.split(",")],
)
parser.add_argument(
    "--thin",
    type=int,
    default=1,
)
parser.add_argument(
    "--test-window",
    type=int,
    default=None,
    help="crop to an NxN window at the densest part, for a timing test",
)

args = parser.parse_args()


# ============================================================== CONFIG =====


DEM_TIF = "data/swissalti_slope/dem_25m_ch.tif"
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
FOS_TIF = f"output/impacts/01_maps/{args.fos_map}"

TARGET_CANTON_IDS = args.canton_ids

FOS_THRESHOLD = 1.0  # a cell releases if FoS < this

THIN = args.thin
RUNOUT_HEADROOM_M = 2000.0

CELL_SIZE_M = 25.0
NODATA = -9999.0
FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
OUTDIR_BASE = f"{FLOWPY_DIR}/flowpy_input/baseline"

# =========================================================== END CONFIG ====


def load_raster(path, label):
    """Read band 1 plus the bits needed to check alignment."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label}: {path} not found.")
    with rasterio.open(path) as src:
        return {
            "data": src.read(1),
            "profile": src.profile,
            "transform": src.transform,
            "nodata": src.nodata,
            "shape": (src.height, src.width),
        }


def check_aligned(a, b, label_a, label_b):
    """Fail loudly if two rasters aren't on the same grid."""
    if a["shape"] != b["shape"]:
        raise ValueError(f"{label_a} {a['shape']} != {label_b} {b['shape']}")
    if not np.allclose(np.array(a["transform"])[:6], np.array(b["transform"])[:6]):
        raise ValueError(f"{label_a} and {label_b} have different transforms.")


def canton_crop_window(canton, target_ids, headroom_m, cell_size, shape):
    """Bounding box of the target canton(s), padded for runout headroom.

    Keyed to the canton so the window is IDENTICAL for every scenario.
    """
    mask = np.isin(canton, target_ids)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        present = sorted(int(v) for v in np.unique(canton) if v > 0)
        raise ValueError(
            f"No pixels for canton ID(s) {target_ids}. Present in raster: {present}"
        )

    pad = int(np.ceil(headroom_m / cell_size))
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, shape[1])
    return r0, r1, c0, c1


def thin_mask(mask, factor):
    """Keep every factor-th row and column of a boolean mask."""
    if factor <= 1:
        return mask
    keep = np.zeros(mask.shape, dtype=bool)
    keep[::factor, ::factor] = True
    return mask & keep


def main():
    os.makedirs(OUTDIR_BASE, exist_ok=True)

    print("=" * 68)
    print(f"EXPORT RELEASE RASTER'")
    print("=" * 68)
    print(f"  FoS map : {FOS_TIF}")
    print(f"  thin    : every {THIN} cell(s)  ({THIN * CELL_SIZE_M:.0f} m spacing)")

    # ---- load ----
    fos_r = load_raster(FOS_TIF, "FoS")
    dem_r = load_raster(DEM_TIF, "DEM")
    canton_r = load_raster(CANTON_ID_TIF, "canton ID")

    check_aligned(fos_r, dem_r, "FoS", "DEM")
    check_aligned(fos_r, canton_r, "FoS", "canton ID")
    print(f"  grids aligned: {fos_r['shape']}")

    fos = fos_r["data"].astype(np.float32)
    dem = dem_r["data"].astype(np.float32)
    canton = canton_r["data"]

    # ---- crop window, keyed to the canton (fixed across scenarios) ----
    if TARGET_CANTON_IDS is None:
        r0, r1, c0, c1 = 0, fos.shape[0], 0, fos.shape[1]
        print("  scope: WHOLE COUNTRY (this will not finish - use a canton)")
    else:
        r0, r1, c0, c1 = canton_crop_window(
            canton, TARGET_CANTON_IDS, RUNOUT_HEADROOM_M, CELL_SIZE_M, fos.shape
        )
        print(f"  scope: canton IDs {TARGET_CANTON_IDS}")

    # ---- optional small window for timing tests ----
    if args.test_window:
        n = args.test_window
        sub = np.isfinite(fos[r0:r1, c0:c1]) & (fos[r0:r1, c0:c1] < FOS_THRESHOLD)
        # centre the window on the row/col with the most unstable cells
        cr = int(np.argmax(sub.sum(axis=1)))
        cc = int(np.argmax(sub.sum(axis=0)))
        r0, r1 = r0 + max(cr - n // 2, 0), r0 + max(cr - n // 2, 0) + n
        c0, c1 = c0 + max(cc - n // 2, 0), c0 + max(cc - n // 2, 0) + n
        r1, c1 = min(r1, fos.shape[0]), min(c1, fos.shape[1])
        print(f"  TEST WINDOW: {n}x{n} at the densest unstable area")

    fos = fos[r0:r1, c0:c1]
    dem = dem[r0:r1, c0:c1]
    canton = canton[r0:r1, c0:c1]
    new_transform = fos_r["transform"] * Affine.translation(c0, r0)
    print(f"  cropped: rows[{r0}:{r1}] cols[{c0}:{c1}] -> {fos.shape}")

    # ---- DEM with Flow-Py's nodata convention ----
    dem_bad = ~np.isfinite(dem)
    if dem_r["nodata"] is not None:
        dem_bad |= dem == dem_r["nodata"]
    dem_out = np.where(dem_bad, NODATA, dem).astype(np.float32)

    # ---- release mask ----
    unstable_all = np.isfinite(fos) & (fos < FOS_THRESHOLD)
    if TARGET_CANTON_IDS is not None:
        unstable_all &= np.isin(canton, TARGET_CANTON_IDS)

    n_all = int(unstable_all.sum())

    # A release cell on DEM nodata is a silent Flow-Py failure.
    on_nodata = int((unstable_all & dem_bad).sum())
    if on_nodata:
        print(f"  [warn] {on_nodata:,} release cells sit on DEM nodata - dropped")
        unstable_all &= ~dem_bad

    release_mask = thin_mask(unstable_all, args.thin)
    n_release = int(release_mask.sum())

    if n_release == 0:
        raise ValueError("No release cells left. Check FOS_THRESHOLD and canton IDs.")

    release = np.zeros(fos.shape, dtype=np.float32)
    release[release_mask] = 1.0

    # ---- report ----
    valid_px = int(np.isfinite(fos).sum())
    print()
    print(
        f"  unstable cells (before thinning) : {n_all:,}"
        f"  ({n_all * CELL_SIZE_M**2 / 1e6:.1f} km2)"
    )
    print(f"  release cells  (after thinning)  : {n_release:,}")
    print(f"  unstable share of valid terrain  : {100 * n_all / max(valid_px, 1):.1f}%")

    # crude runtime hint - calibrate the constant from your own timing test
    print(f"\n  Flow-Py runtime scales with release-cell count.")
    print(f"  Run --test-window 300 --thin {args.thin} first and time it, then:")
    print(f"      full_runtime = test_runtime x ({n_release} / test_release_cells)")

    # ---- write ----
    p = fos_r["profile"].copy()
    p.update(
        driver="GTiff",
        dtype="float32",
        nodata=NODATA,
        height=fos.shape[0],
        width=fos.shape[1],
        transform=new_transform,
        count=1,
        compress="lzw",
    )

    print()
    for name, arr in [("dem", dem_out), ("release", release)]:
        path = f"{OUTDIR_BASE}/{name}.tif"
        with rasterio.open(path, "w", **p) as dst:
            dst.write(arr, 1)
        print(f"  -> {path}")

    # ---- metadata: needed to paste results back onto the national grid ----
    meta = {
        "fos_source": FOS_TIF,
        "fos_threshold": FOS_THRESHOLD,
        "thin_factor": THIN,
        "release_spacing_m": THIN * CELL_SIZE_M,
        "target_canton_ids": TARGET_CANTON_IDS,
        "crop_window": [r0, r1, c0, c1],
        "shape": list(fos.shape),
        "n_unstable_cells": n_all,
        "n_release_cells": n_release,
        "unstable_area_km2": round(n_all * CELL_SIZE_M**2 / 1e6, 2),
    }
    meta_path = f"{OUTDIR_BASE}/meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  -> {meta_path}")


if __name__ == "__main__":
    main()
