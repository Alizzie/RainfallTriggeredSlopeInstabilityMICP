"""
unstable_by_region.py - What share of each drought region's soil-mantled terrain
is unstable (FoS < 1), using the Phase 2 fos_root.tif map.


Output:
    output/impacts/03_regional/unstable_share_by_region.csv
    output/impacts/03_regional/unstable_share_by_region.png
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import region_map as rmap

FOS_TIF = "output/impacts/01_maps/fos_bare.tif"
OUTDIR = "output/impacts/04_regional"
os.makedirs(OUTDIR, exist_ok=True)

FOS_THRESHOLD = 1.0
MIN_PIXELS = 50  # regions with fewer valid soil pixels than this get flagged,
# not silently included - percentages on tiny samples mislead


def main():
    if not os.path.exists(FOS_TIF):
        raise FileNotFoundError(f"{FOS_TIF} not found - run root_cohesion.py first.")

    with rasterio.open(FOS_TIF) as src:
        fos = src.read(1).astype(np.float32)
        transform = src.transform
        shape = fos.shape

    valid = np.isfinite(fos)
    unstable = valid & (fos < FOS_THRESHOLD)

    # Rasterize every drought region onto the SAME grid as the FoS map
    geometries, label_points, ids = dl.load_regions()
    shapes = [(geometries[rid], int(rid)) for rid in ids]
    region_raster = rasterize(
        shapes=shapes, out_shape=shape, transform=transform, fill=0, dtype=np.int32
    )

    present_ids = sorted(set(np.unique(region_raster)) - {0})
    missing = set(ids) - set(present_ids)
    print(f"Regions present on the FoS grid: {len(present_ids)}/{len(ids)}")
    if missing:
        print(
            f"  missing (too small / outside 15-45deg soil mask entirely): {sorted(missing)}"
        )

    rows = []
    for rid in present_ids:
        in_region = region_raster == rid
        n_valid = int((in_region & valid).sum())
        n_unstable = int((in_region & unstable).sum())

        if n_valid < MIN_PIXELS:
            rows.append(
                {
                    "region_id": rid,
                    "n_valid_pixels": n_valid,
                    "n_unstable_pixels": n_unstable,
                    "unstable_pct": np.nan,
                    "flag": f"too few pixels (<{MIN_PIXELS})",
                }
            )
            continue

        rows.append(
            {
                "region_id": rid,
                "n_valid_pixels": n_valid,
                "n_unstable_pixels": n_unstable,
                "unstable_pct": round(100 * n_unstable / n_valid, 2),
                "flag": "",
            }
        )

    df = pd.DataFrame(rows).sort_values("unstable_pct", ascending=False)
    csv_path = f"{OUTDIR}/unstable_share_by_region.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nTop 10 regions by unstable share:")
    print(df.head(10).to_string(index=False))
    print(f"\nBottom 10 regions by unstable share:")
    print(df.dropna(subset=["unstable_pct"]).tail(10).to_string(index=False))

    # ---- Choropleth ----
    values = dict(zip(df["region_id"], df["unstable_pct"]))
    fig, ax = plt.subplots(figsize=(12, 9))
    fig, ax, mappable = rmap.plot_region_choropleth(
        values,
        geometries,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        label_points=label_points,
        label_ids=True,
        title="Unstable Soil-Mantled Area by Region (FoS < 1, with roots)",
        cbar_label="% of valid pixels unstable",
        axis_off=True,
    )
    fig.tight_layout()
    png_path = f"{OUTDIR}/unstable_share_by_region.png"
    fig.savefig(png_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {csv_path}")
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
