"""
06_cantonal_unstable_regions.py - Canton version of unstable_by_region.py.

Output:
    output/impacts/06_cantonal_unstable/unstable_share_by_canton.csv
    output/impacts/06_cantonal_unstable/unstable_share_by_canton.png
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOS_TIF = "output/impacts/01_maps/fos_bare.tif"
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
CANTON_NAMES_CSV = "data/swissboundaries/canton_names.csv"

OUTDIR = "output/impacts/06_cantonal_unstable"
os.makedirs(OUTDIR, exist_ok=True)

FOS_THRESHOLD = 1.0
MIN_PIXELS = 50  # cantons with fewer valid soil pixels than this get flagged,
# not silently included - percentages on tiny samples mislead


def load_canton_names():
    if os.path.exists(CANTON_NAMES_CSV):
        names_df = pd.read_csv(CANTON_NAMES_CSV)
        return dict(zip(names_df["KANTONSNUM"], names_df["NAME"]))
    return {}


def build_metric_map(canton_id_raster, canton_df, metric_col):
    """Paint every pixel with its enclosing canton's value - a choropleth
    without needing canton vector geometries."""
    max_id = int(canton_id_raster.max())
    lookup = np.full(max_id + 1, np.nan, dtype=np.float32)
    for _, row in canton_df.iterrows():
        cid = int(row["canton_id"])
        if 0 <= cid <= max_id:
            lookup[cid] = row[metric_col]
    idx = np.clip(canton_id_raster, 0, max_id)
    metric_map = lookup[idx]
    metric_map[canton_id_raster <= 0] = np.nan
    return metric_map


def main():
    if not os.path.exists(FOS_TIF):
        raise FileNotFoundError(f"{FOS_TIF} not found - run root_cohesion.py first.")
    if not os.path.exists(CANTON_ID_TIF):
        raise FileNotFoundError(
            f"{CANTON_ID_TIF} not found - build it first (see QGIS_prep_guide.md, section 4)."
        )

    with rasterio.open(FOS_TIF) as src:
        fos = src.read(1).astype(np.float32)
        shape = fos.shape

    with rasterio.open(CANTON_ID_TIF) as src:
        canton_id = src.read(1).astype(np.int16)

    if canton_id.shape != shape:
        raise ValueError(f"canton_id grid {canton_id.shape} != fos grid {shape}")

    canton_names = load_canton_names()

    valid = np.isfinite(fos)
    unstable = valid & (fos < FOS_THRESHOLD)

    present_ids = sorted(int(c) for c in np.unique(canton_id) if c > 0)
    print(f"Cantons present on the FoS grid: {len(present_ids)}")

    rows = []
    for cid in present_ids:
        in_canton = canton_id == cid
        n_valid = int((in_canton & valid).sum())
        n_unstable = int((in_canton & unstable).sum())

        if n_valid < MIN_PIXELS:
            rows.append(
                {
                    "canton_id": cid,
                    "canton_name": canton_names.get(cid, ""),
                    "n_valid_pixels": n_valid,
                    "n_unstable_pixels": n_unstable,
                    "unstable_pct": np.nan,
                    "flag": f"too few pixels (<{MIN_PIXELS})",
                }
            )
            continue

        rows.append(
            {
                "canton_id": cid,
                "canton_name": canton_names.get(cid, ""),
                "n_valid_pixels": n_valid,
                "n_unstable_pixels": n_unstable,
                "unstable_pct": round(100 * n_unstable / n_valid, 2),
                "flag": "",
            }
        )

    df = pd.DataFrame(rows).sort_values("unstable_pct", ascending=False)
    csv_path = f"{OUTDIR}/unstable_share_by_canton.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nTop 10 cantons by unstable share:")
    print(df.head(10).to_string(index=False))
    print(f"\nBottom 10 cantons by unstable share:")
    print(df.dropna(subset=["unstable_pct"]).tail(10).to_string(index=False))

    # ---- choropleth, painted straight from the canton_id raster ----
    metric_map = build_metric_map(canton_id, df, "unstable_pct")

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(metric_map, cmap="YlOrRd", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("% of valid soil pixels unstable (FoS < 1, with roots)")
    ax.set_title("Unstable Soil-Mantled Area by Canton", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png_path = f"{OUTDIR}/unstable_share_by_canton.png"
    fig.savefig(png_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {csv_path}")
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
