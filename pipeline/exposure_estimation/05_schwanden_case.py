"""
05_schwanden_case.py - Does the model flag the Wagenrunse slope?

THE CASE
Coordinates 724921 / 205698 (LV03) appear five times in the WSL record for
Glarus: December 2020, then April, May, 20 August and 29 August 2023. The last
was a collapse - roughly 30,000 m3 released, 38 properties buried or destroyed,
112 people evacuated, two company buildings lost. Remediation after the 2020
slide had already cost CHF 3.9 million.

It is the largest recent landslide in the Swiss record, it repeated at the same
spot, and it is in a canton the model was run over blind. So it is the single
best test of whether the hazard map identifies a real problem slope.

The December 2020 entry also notes that protective forest (Schutzwald) had to
be cleared to allow drainage work - so this site sits at the intersection of
the infrastructure and conservation arguments.

Outputs:
    schwanden_case.txt   the numbers
    schwanden_case.png   zoomed map of the site
"""

import json
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch

from core import data_loader as dl

argparse = argparse.ArgumentParser()
argparse.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)

args = argparse.parse_args()

FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"
OUTDIR = FLOWPY_DIR + "/validation"
os.makedirs(OUTDIR, exist_ok=True)

# Wagenrunse / Niederentalstrasse, Schwanden. LV03 as recorded by WSL.
SITE_NAME = "Schwanden - Wagenrunse"
SITE_X_LV03 = 724921
SITE_Y_LV03 = 205698

ZOOM_M = 1500
SEARCH_RADII_M = [100, 250, 500, 1000]

CELL_SIZE_M = 25.0

LAYERS = {
    "roads": "data/infra/roads_binary_25m.tif",
    "buildings": "data/infra/buildings_id_25m.tif",
    "schutzwald": "data/infra/schutzwald_rutschungen_25m.tif",
}

# =========================================================== END CONFIG ====


def crop_window():
    meta = json.load(open(f"{INPUT_DIR}/meta.json"))
    r0, r1, c0, c1 = meta["crop_window"]
    return Window(c0, r0, c1 - c0, r1 - r0)


def load_all():
    with rasterio.open(f"{FLOWPY_DIR}/z_delta.tif") as src:
        z = src.read(1).astype(np.float32)
        transform = src.transform
    with rasterio.open(f"{FLOWPY_DIR}/cell_counts.tif") as src:
        counts = src.read(1).astype(np.float32)
    with rasterio.open(f"{INPUT_DIR}/dem.tif") as src:
        dem = src.read(1).astype(np.float32)
    with rasterio.open(f"{INPUT_DIR}/release.tif") as src:
        release = src.read(1) > 0

    dem = np.where(dem <= -9000, np.nan, dem)

    win = crop_window()
    layers = {}
    for name, path in LAYERS.items():
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        with rasterio.open(path) as src:
            layers[name] = src.read(1, window=win) > 0

    return z, counts, dem, release, layers, transform


def disc(shape, row, col, radius_m, cell_size=CELL_SIZE_M):
    """Boolean mask of cells within radius_m of (row, col)."""
    rad = int(np.ceil(radius_m / cell_size))
    h, w = shape
    r0, r1 = max(row - rad, 0), min(row + rad + 1, h)
    c0, c1 = max(col - rad, 0), min(col + rad + 1, w)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    d = np.hypot((yy - row) * cell_size, (xx - col) * cell_size)
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = d <= radius_m
    return m


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    z, counts, dem, release, layers, transform = load_all()
    footprint = np.isfinite(z) & (z > 0)

    x, y = dl.to_lv95(SITE_X_LV03, SITE_Y_LV03)
    row, col = rasterio.transform.rowcol(transform, x, y)
    h, w = footprint.shape
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError(f"{SITE_NAME} falls outside the modelled grid.")

    log("=" * 66)
    log(f"CASE STUDY: {SITE_NAME}")
    log("=" * 66)
    log(f"  LV03 {SITE_X_LV03} / {SITE_Y_LV03}  ->  LV95 {x:.0f} / {y:.0f}")
    log(f"  grid cell (row {row}, col {col}), elevation {dem[row, col]:.0f} m")
    log("")

    # --- the headline question ---
    inside = bool(footprint[row, col])
    log(
        f"  Damage site inside a modelled runout corridor : {'YES' if inside else 'no'}"
    )
    if inside:
        log(f"    paths converging on this cell            : {counts[row, col]:.0f}")
        log(f"    z_delta                                  : {z[row, col]:.1f} m")

    # --- release cells upslope ---
    log("")
    log(f"  {'radius':>8} {'release cells':>14} {'corridor cells':>15}")
    for r in SEARCH_RADII_M:
        d = disc(footprint.shape, row, col, r)
        log(
            f"  {r:8.0f} {int((release & d).sum()):14,} "
            f"{int((footprint & d).sum()):15,}"
        )

    # --- what else is in the corridor here ---
    log("")
    near = disc(footprint.shape, row, col, 500)
    for name, layer in layers.items():
        hit = int((footprint & near & layer).sum())
        total = int((near & layer).sum())
        if total:
            log(f"  {name:12s} within 500 m: {hit:,}/{total:,} cells in a corridor")

    # --- verdict ---
    log("")
    if inside:
        log("  The model independently flags the slope that collapsed in 2023,")
        log("  from rainfall and terrain alone, with no knowledge of the event.")
    else:
        d500 = disc(footprint.shape, row, col, 500)
        if (footprint & d500).any():
            log("  The site itself is not flagged, but corridors reach within")
            log("  500 m. Given that WSL coordinates are approximate, this is a")
            log("  partial hit - report it as such rather than as a clean one.")
        else:
            log("  The model does not flag this site. Worth reporting honestly:")
            log("  the Wagenrunse failure began as a rock collapse feeding a")
            log("  loose-material slide, which is outside the shallow-soil")
            log("  physics this model covers.")

    # --- map ---
    rad = int(ZOOM_M / CELL_SIZE_M)
    r0, r1 = max(row - rad, 0), min(row + rad + 1, h)
    c0, c1 = max(col - rad, 0), min(col + rad + 1, w)

    sub_dem = dem[r0:r1, c0:c1]
    filled = np.where(np.isfinite(sub_dem), sub_dem, np.nanmin(sub_dem))
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(filled, vert_exag=1.5, dx=CELL_SIZE_M, dy=CELL_SIZE_M)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(shade, cmap="gray", vmin=0, vmax=2.0)

    def paint(mask, colour, alpha):
        sub = mask[r0:r1, c0:c1]
        if sub.any():
            ax.imshow(
                np.ma.masked_where(~sub, np.ones(sub.shape, dtype=np.uint8)),
                cmap=ListedColormap([colour]),
                alpha=alpha,
                interpolation="nearest",
            )

    paint(footprint, "#ff6d00", 0.55)
    paint(release, "#1a237e", 0.55)
    if "schutzwald" in layers:
        paint(layers["schutzwald"], "#2e7d32", 0.45)
    if "roads" in layers:
        paint(layers["roads"], "#ffffff", 0.85)

    ax.scatter(
        [col - c0],
        [row - r0],
        s=220,
        marker="*",
        c="#d50000",
        edgecolor="black",
        linewidth=1.0,
        zorder=6,
        label="recorded damage site",
    )

    # contours for context
    levels = np.arange(
        np.floor(np.nanmin(sub_dem) / 100) * 100, np.nanmax(sub_dem) + 100, 100
    )
    if len(levels) > 1:
        ax.contour(
            np.where(np.isfinite(sub_dem), sub_dem, np.nan),
            levels=levels,
            colors="#37474f",
            linewidths=0.4,
            alpha=0.5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Patch(facecolor="#1a237e", alpha=0.55, label="release cells (FoS < 1)"),
        Patch(facecolor="#ff6d00", alpha=0.55, label="modelled runout corridor"),
    ]
    if "schutzwald" in layers:
        handles.append(Patch(facecolor="#2e7d32", alpha=0.45, label="Schutzwald"))
    if "roads" in layers:
        handles.append(Patch(facecolor="#ffffff", alpha=0.85, label="roads"))
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_title(
        f"Does the modelled runout reach the damage site in {SITE_NAME} ",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    png = f"{OUTDIR}/schwanden_case.png"
    fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log("")
    log(f"-> {png}")

    txt = f"{OUTDIR}/schwanden_case.txt"
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"-> {txt}")


if __name__ == "__main__":
    main()
