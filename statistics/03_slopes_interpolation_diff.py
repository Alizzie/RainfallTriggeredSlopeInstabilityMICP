#!/usr/bin/env python3
"""
Statistics + histogram for a slope-difference raster (e.g. slope_diff_ch.tif).

Reads the raster, strips NoData / non-finite cells, prints summary statistics,
and saves a two-panel histogram (linear + log y) binned across the full
min-to-max range so both the central spike and the rare steep-terrain tails
are visible.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

# ---------------------------------------------------------------------------
# Config (override from the command line if you like)
# ---------------------------------------------------------------------------
RASTER = sys.argv[1] if len(sys.argv) > 1 else "data/swissalti_slope/slope_diff_ch.tif"
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUTPUT_DIR = "output/statistics/02_interpolation_diff"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PNG = f"{OUTPUT_DIR}/slope_diff_histogram.png"
OUT_TXT = f"{OUTPUT_DIR}/slope_diff_stats.txt"

# ---------------------------------------------------------------------------
# Read raster, drop NoData and non-finite cells
# ---------------------------------------------------------------------------
with rasterio.open(RASTER) as src:
    arr = src.read(1).astype("float64")  # band 1
    nodata = src.nodata

mask = np.isfinite(arr)  # remove inf / nan
if nodata is not None:
    mask &= arr != nodata  # remove flagged NoData
vals = arr[mask]

if vals.size == 0:
    sys.exit("No valid cells found after masking NoData.")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
stats = {
    "valid cells": vals.size,
    "min": vals.min(),
    "max": vals.max(),
    "range": vals.max() - vals.min(),
    "mean": vals.mean(),
    "std dev": vals.std(),
    "median": np.median(vals),
    "1st percentile": np.percentile(vals, 1),
    "5th percentile": np.percentile(vals, 5),
    "25th percentile": np.percentile(vals, 25),
    "75th percentile": np.percentile(vals, 75),
    "95th percentile": np.percentile(vals, 95),
    "99th percentile": np.percentile(vals, 99),
    "% within +/-0.5": 100.0 * np.mean(np.abs(vals) <= 0.5),
    "% within +/-1.0": 100.0 * np.mean(np.abs(vals) <= 1.0),
    "% within +/-2.0": 100.0 * np.mean(np.abs(vals) <= 2.0),
}

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"\nSlope-difference statistics  ({RASTER})\n")
    f.write("-" * 60 + "\n")
    for k, v in stats.items():
        if k == "valid cells":
            f.write(f"  {k:<18}: {v:,} \n")
        elif k.startswith("%"):
            f.write(f"  {k:<18}: {v:6.2f} \n%")
        else:
            f.write(f"  {k:<18}: {v:10.4f}\n")
    f.write("-" * 60)

# ---------------------------------------------------------------------------
# Histogram: bins from min to max, cell count per bin
# ---------------------------------------------------------------------------
bins = np.linspace(vals.min(), vals.max(), N_BINS + 1)
counts, edges = np.histogram(vals, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])
width = edges[1] - edges[0]

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))

# Linear y — shows the central spike
ax_lin.bar(centers, counts, width=width, color="#c0392b", edgecolor="none")
ax_lin.axvline(0, color="black", lw=0.8)
ax_lin.set_title("Linear scale")
ax_lin.set_xlabel("slope difference (degrees)")
ax_lin.set_ylabel("number of cells")

# Log y — reveals the rare steep-terrain tails
ax_log.bar(centers, counts, width=width, color="#c0392b", edgecolor="none")
ax_log.axvline(0, color="black", lw=0.8)
ax_log.set_yscale("log")
ax_log.set_title("Log scale (reveals tails)")
ax_log.set_xlabel("slope difference (degrees)")
ax_log.set_ylabel("number of cells (log)")

fig.suptitle(
    "Slope difference: Average vs Nearest-Neighbour resampling (25 m)",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\nHistogram saved to: {OUT_PNG}\n")
