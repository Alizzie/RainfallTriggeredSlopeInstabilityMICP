"""
map_wsl_events_by_region.py - WSL landslide events plotted on a regional map,
shaded by how many events each region contains.

Output:
    output/data_analysis/09_wsl_mapping/wsl_events_by_drought_region.png (or _canton.png)
    output/data_analysis/09_wsl_mapping/events_per_drought_region.csv (or _canton.csv)
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation

from core import data_loader as dl

# ============================================================== CONFIG =====

MODE = "drought"  # "drought" or "canton"

# --- drought-region mode ---
MAX_SNAP_M = 2000.0

OUTDIR = "output/data_analysis/09_wsl_mapping"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================== END CONFIG ====


def canton_at(x, y, canton_id, transform):
    row, col = rasterio.transform.rowcol(transform, x, y)
    if row < 0 or row >= canton_id.shape[0] or col < 0 or col >= canton_id.shape[1]:
        return None
    val = int(canton_id[row, col])
    return val if val > 0 else None


def build_metric_map(canton_id_raster, values_by_id):
    """Paint every pixel with its enclosing canton's value - a choropleth
    without needing canton vector geometries."""
    max_id = int(canton_id_raster.max())
    lookup = np.full(max_id + 1, np.nan, dtype=np.float32)
    for cid, val in values_by_id.items():
        if 0 <= cid <= max_id:
            lookup[cid] = val
    idx = np.clip(canton_id_raster, 0, max_id)
    metric_map = lookup[idx]
    metric_map[canton_id_raster <= 0] = np.nan
    return metric_map


def run_drought_mode():
    inv = dl.load_wsl_usable_inventory()
    geometries, label_points, ids = dl.load_regions()

    print(f"{len(inv)} events, {len(ids)} drought regions loaded.")

    region_ids = [
        dl.assign_region(x, y, max_snap_m=MAX_SNAP_M)
        for x, y in zip(inv["x"], inv["y"])
    ]
    inv = inv.assign(region_id=region_ids)

    n_unassigned = inv["region_id"].isna().sum()
    if n_unassigned:
        print(
            f"  {n_unassigned} events could not be assigned to any region (>{MAX_SNAP_M:.0f} m away)"
        )

    counts = inv["region_id"].value_counts().to_dict()
    counts_full = {rid: counts.get(rid, 0) for rid in ids}

    counts_df = (
        pd.Series(counts_full, name="n_events")
        .rename_axis("region_id")
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    counts_df.to_csv(f"{OUTDIR}/events_per_drought_region.csv", index=False)
    print(f"\nTop 10 drought regions by event count:")
    print(counts_df.head(10).to_string(index=False))

    from core import region_map as rmap  # vector-geometry choropleth backend

    fig, ax = plt.subplots(figsize=(12, 9))
    fig, ax, mappable = rmap.plot_region_choropleth(
        counts_full,
        geometries,
        ax=ax,
        cmap="YlOrRd",
        label_points=label_points,
        label_ids=True,
        title="WSL Landslide Events by Drought Region",
        cbar_label="Number of events",
        axis_off=True,
    )

    assigned = inv.dropna(subset=["region_id"])
    ax.scatter(
        assigned["x"],
        assigned["y"],
        s=10,
        facecolor="black",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.55,
        zorder=5,
        label=f"WSL events (n={len(assigned):,})",
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = f"{OUTDIR}/wsl_events_by_drought_region.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {out_path}")
    print(f"-> {OUTDIR}/events_per_drought_region.csv")


def detect_borders(id_raster, thickness_px=4):
    """Border pixels: anywhere a neighbor (N/S/E/W) has a different ID.
    Since there are no vector polygons for cantons, this derives border
    LINES directly from the raster itself.

    Dilated by thickness_px: a raw 1-pixel-wide border (native 25m
    resolution) gets crushed to sub-pixel width once matplotlib downsamples
    a ~14,000-pixel-wide national array down to a ~1,600px PNG, and
    disappears entirely. Thickening it first keeps it visible after that
    downsampling."""
    border = np.zeros_like(id_raster, dtype=bool)
    diff_v = id_raster[:-1, :] != id_raster[1:, :]
    border[:-1, :] |= diff_v
    border[1:, :] |= diff_v
    diff_h = id_raster[:, :-1] != id_raster[:, 1:]
    border[:, :-1] |= diff_h
    border[:, 1:] |= diff_h
    return binary_dilation(border, iterations=thickness_px)


def crop_to_data_bbox(id_raster, transform):
    """Tight-crop to the actual nonzero footprint (Switzerland's real shape),
    not the full padded national raster extent - matches how the
    drought-region plot autoscales to its geometries' real extent instead of
    an arbitrary bounding rectangle."""
    from rasterio.windows import Window, bounds as window_bounds

    rows, cols = np.nonzero(id_raster > 0)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    cropped = id_raster[r0:r1, c0:c1]
    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    left, bottom, right, top = window_bounds(window, transform)
    return cropped, (left, right, bottom, top)


def run_canton_mode():
    inv = dl.load_wsl_usable_inventory()
    canton_id, transform, bounds = dl.load_canton_raster()
    canton_names = dl.load_canton_names()
    present_ids = sorted(int(c) for c in np.unique(canton_id) if c > 0)
    print(f"{len(inv)} events, {len(present_ids)} cantons present in raster.")

    region_ids = [
        canton_at(x, y, canton_id, transform) for x, y in zip(inv["x"], inv["y"])
    ]
    inv = inv.assign(canton_id=region_ids)

    n_unassigned = inv["canton_id"].isna().sum()
    if n_unassigned:
        print(f"  {n_unassigned} events fell outside all canton polygons")

    counts = inv["canton_id"].value_counts().to_dict()
    counts_full = {cid: counts.get(cid, 0) for cid in present_ids}

    counts_df = (
        pd.Series(counts_full, name="n_events").rename_axis("canton_id").reset_index()
    )
    counts_df["canton_name"] = counts_df["canton_id"].map(canton_names)
    counts_df = counts_df.sort_values("n_events", ascending=False)
    counts_df.to_csv(f"{OUTDIR}/events_per_canton.csv", index=False)
    print(f"\nTop 10 cantons by event count:")
    print(counts_df.head(10).to_string(index=False))

    # ---- tight-crop to Switzerland's actual footprint, not the padded national extent ----
    canton_id_cropped, (left, right, bottom, top) = crop_to_data_bbox(
        canton_id, transform
    )
    border_mask = detect_borders(canton_id_cropped)

    # ---- choropleth painted directly from the raster, no geometries needed ----
    metric_map = build_metric_map(canton_id_cropped, counts_full)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(
        metric_map,
        cmap="YlOrRd",
        interpolation="nearest",
        extent=[left, right, bottom, top],
    )

    # overlay canton borders as black lines (derived from the raster, since
    # there are no vector polygons to draw outlines from directly)
    border_display = np.ma.masked_where(
        ~border_mask, np.ones_like(border_mask, dtype=float)
    )
    ax.imshow(
        border_display,
        cmap=ListedColormap(["black"]),
        extent=[left, right, bottom, top],
        interpolation="nearest",
        zorder=3,
        alpha=0.8,
    )

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Number of WSL events")

    assigned = inv.dropna(subset=["canton_id"])
    ax.scatter(
        assigned["x"],
        assigned["y"],
        s=10,
        facecolor="black",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.55,
        zorder=5,
        label=f"WSL events (n={len(assigned):,})",
    )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("WSL Landslide Events by Canton", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = f"{OUTDIR}/wsl_events_by_canton.png"
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {out_path}")
    print(f"-> {OUTDIR}/events_per_canton.csv")


def main():
    if MODE == "drought":
        run_drought_mode()
    elif MODE == "canton":
        run_canton_mode()
    else:
        raise ValueError(f"Unknown MODE '{MODE}' - use 'drought' or 'canton'.")


if __name__ == "__main__":
    main()
