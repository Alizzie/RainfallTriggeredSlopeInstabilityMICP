"""
map_wsl_events_by_region.py - WSL landslide events plotted on the drought-region
map, with each region shaded by how many events it contains.

Uses region_map.py's choropleth (borders + fill) + a scatter overlay of the raw
event points, so you can see both the regional pattern (fill) and the actual
point distribution (dots) in one figure.

Output:
    output/impacts/03_regional/wsl_events_by_region.png
    output/impacts/03_regional/events_per_region.csv
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import region_map as rmap

OUTDIR = "output/data_analysis/09_wsl_mapping"
os.makedirs(OUTDIR, exist_ok=True)

MAX_SNAP_M = 2000.0  # matches dl.get_region_params default; events just outside
# a region boundary still get assigned rather than dropped


def main():
    inv = dl.load_wsl_usable_inventory()
    geometries, label_points, ids = dl.load_regions()

    print(f"{len(inv)} events, {len(ids)} drought regions loaded.")

    # Assign every event to a region (point-in-polygon, bounded snap for edge cases)
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
    # every region gets a value, even 0, so the choropleth doesn't leave gaps grey-by-default
    counts_full = {rid: counts.get(rid, 0) for rid in ids}

    counts_df = (
        pd.Series(counts_full, name="n_events")
        .rename_axis("region_id")
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    counts_df.to_csv(f"{OUTDIR}/events_per_region.csv", index=False)
    print(f"\nTop 10 regions by event count:")
    print(counts_df.head(10).to_string(index=False))

    # ---- Plot: choropleth fill (event count) + scatter overlay (raw points) ----
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
    out_path = f"{OUTDIR}/wsl_events_by_region.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {out_path}")
    print(f"-> {OUTDIR}/events_per_region.csv")


if __name__ == "__main__":
    main()
