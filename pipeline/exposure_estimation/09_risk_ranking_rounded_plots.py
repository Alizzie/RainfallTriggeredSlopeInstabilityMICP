"""

09_risk_ranking.py - spatial ranking of expected annual landslide damage.

"""

import json

import os

import sys

import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

import pandas as pd

import rasterio

from rasterio.windows import Window

from scipy import ndimage

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import LightSource, LogNorm

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


def format_chf_thousands(value, _position=None):
    """Format displayed CHF amounts rounded to the nearest CHF 1,000."""
    return f"{round(value / 1000) * 1000:,.0f}"


argparse = argparse.ArgumentParser()

argparse.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)

argparse.add_argument(
    "--canton_ids",
    default=[15],
    type=lambda s: [int(x) for x in s.split(",")],
)

argparse.add_argument(
    "--canton_label",
    default="Glarus",
    help="Label for the canton",
)

args = argparse.parse_args()

# ============================================================== CONFIG =====

FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"

CELL_COUNTS_TIF = f"{FLOWPY_DIR}/cell_counts.tif"

INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"

DEM_TIF = f"{INPUT_DIR}/dem.tif"

META_JSON = f"{INPUT_DIR}/meta.json"

map_type = next(
    t for t in ["root_micp", "root", "micp", "bare"] if args.flowpy_name.startswith(t)
)

bare_name = (
    "bare_" + args.flowpy_name.split("_")[-1] if "_" in args.flowpy_name else map_type
)

EAD_CSV = f"output/flowpy/glarus/{bare_name}/ead/ead_by_asset.csv"

EAD_MEAN_COLNAME = map_type + "_ead_mean" if map_type != "bare" else "ead_mean"

# Keep ranking outputs separate from exposure/EAD outputs.

OUTDIR = f"{FLOWPY_DIR}/risk_ranking"

os.makedirs(OUTDIR, exist_ok=True)

CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"

TARGET_CANTON_IDS = args.canton_ids

CANTON_LABEL = args.canton_label

CELL_SIZE_M = 25.0

TOP_N_SEGMENTS = 20

MAP_TOP_N = 10  # number of highest-ranked locations to annotate on maps

TOP_RISK_TARGETS = 50  # number of ranked locations written to Flow-Py target rasters

ASSETS = {
    "road": {
        "raster": "data/infra/roads_binary_25m.tif",
        "display": "Road",
        "abbr": "Rd",
    },
    "railway": {
        "raster": "data/infra/railway_binary_25m.tif",
        "display": "Railway",
        "abbr": "Rl",
    },
    "building": {
        "raster": "data/infra/buildings_id_25m.tif",
        "display": "Building",
        "abbr": "Bld",
    },
    "forest": {
        "raster": "data/swissTLM3D/tlm_forest_25m_ch.tif",
        "display": "Forest",
        "abbr": "For",
    },
    "agricultural land": {
        "raster": "data/infra/nutzungsgebiete.tif",
        "display": "Agriculture",
        "abbr": "Agr",
    },
}

# =========================================================== END CONFIG ====


def require_file(path, label):

    if not os.path.exists(path):

        raise FileNotFoundError(f"{label} not found: {path}")


def load_grid():
    """Load Flow-Py convergence, DEM, canton mask, transform and crop window."""

    require_file(CELL_COUNTS_TIF, "cell_counts raster")

    require_file(DEM_TIF, "DEM")

    require_file(META_JSON, "Flow-Py input metadata")

    require_file(CANTON_ID_TIF, "canton raster")

    with rasterio.open(CELL_COUNTS_TIF) as src:

        counts = src.read(1).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform

    with rasterio.open(DEM_TIF) as src:

        dem = src.read(1).astype(np.float32)

    # DEM and cell_counts should refer to the same cropped Flow-Py grid.

    if dem.shape != counts.shape:

        raise ValueError(
            f"DEM shape {dem.shape} != cell_counts shape {counts.shape}. "
            "Check INPUT_DIR for the selected Flow-Py run."
        )

    dem = np.where(dem <= -9000, np.nan, dem)

    with open(META_JSON, "r", encoding="utf-8") as fh:

        meta = json.load(fh)

    r0, r1, c0, c1 = meta["crop_window"]

    win = Window(c0, r0, c1 - c0, r1 - r0)

    with rasterio.open(CANTON_ID_TIF) as src:

        canton = src.read(1, window=win)

    if canton.shape != counts.shape:

        raise ValueError(
            f"Canton crop shape {canton.shape} != cell_counts shape {counts.shape}. "
            "Check META_JSON / INPUT_DIR for the selected Flow-Py run."
        )

    in_canton = np.isin(canton, TARGET_CANTON_IDS)

    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

    counts[counts < 0] = 0

    return counts, dem, in_canton, transform, profile, win


def load_asset(path, win, ref_shape, label):
    """Read one infrastructure raster using the same crop window as Flow-Py."""

    if not os.path.exists(path):

        print(f"  [skip] {label}: asset raster not found: {path}")

        return None

    with rasterio.open(path) as src:

        arr = src.read(1, window=win)

    if arr.shape != ref_shape:

        raise ValueError(f"{label} shape {arr.shape} != Flow-Py grid {ref_shape}")

    return arr > 0


def load_ead():
    """Read final empirical asset EAD from 06_compute_ead.py output."""

    require_file(EAD_CSV, "EAD table")

    df = pd.read_csv(EAD_CSV)

    required = {"asset", f"{EAD_MEAN_COLNAME}"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"{EAD_CSV} is missing required columns: {sorted(missing)}")

    df["asset"] = df["asset"].astype(str).str.strip().str.lower()

    df[f"{EAD_MEAN_COLNAME}"] = pd.to_numeric(
        df[f"{EAD_MEAN_COLNAME}"], errors="coerce"
    )

    # These are optional but useful in the exported audit table/report.

    for col in ["events", "events_per_year", "cost_median"]:

        if col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="asset", keep="first").set_index("asset")

    return df


# ------------------------------------------------------------- risk allocation


def risk_for_asset(counts, asset_mask, in_canton, asset_ead):
    """Allocate one asset's fixed EAD over its Flow-Py-reached cells.

    The total is intentionally conserved:

        sum(risk_i) = asset_ead

    whenever at least one asset cell has positive cell_counts.

    """

    w = np.where(asset_mask & in_canton, counts, 0.0).astype(np.float64)

    total_weight = float(w.sum())

    if total_weight <= 0 or not np.isfinite(asset_ead) or asset_ead <= 0:

        return np.zeros_like(w, dtype=np.float32), total_weight

    field = (w / total_weight) * float(asset_ead)

    return field.astype(np.float32), total_weight


def rank_segments(risk, transform, top_n=TOP_N_SEGMENTS):
    """Group contiguous risk-bearing cells and rank clusters by CHF/year."""

    mask = risk > 0

    if not mask.any():

        return pd.DataFrame()

    labels, n_groups = ndimage.label(mask, structure=np.ones((3, 3)))

    group_ids = np.arange(1, n_groups + 1)

    sums = ndimage.sum(risk, labels, index=group_ids)

    sizes = ndimage.sum(mask, labels, index=group_ids)

    centres = ndimage.center_of_mass(risk, labels, index=group_ids)

    order = np.argsort(sums)[::-1]

    if top_n is not None:

        order = order[:top_n]

    rows = []

    for rank, idx in enumerate(order, start=1):

        r, c = centres[idx]

        x, y = rasterio.transform.xy(transform, int(round(r)), int(round(c)))

        rows.append(
            {
                "rank": rank,
                "risk_chf_per_year": round(float(sums[idx]), 0),
                "cells": int(sizes[idx]),
                # Approximate linear extent only; meaningful mainly for linear
                # assets. Keep cells as the primary size measure.
                "approx_cell_extent_m": round(float(sizes[idx]) * CELL_SIZE_M, 0),
                "x_lv95": round(float(x), 0),
                "y_lv95": round(float(y), 0),
                "cluster_id": int(group_ids[idx]),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figures


def plot_risk_map(risk, dem, in_canton, transform, outpath, title_label=None, top=None):
    """Plot CHF/year per cell over hillshade.

    If title_label is provided, the figure is for one asset only.

    Otherwise it is the combined mapped-asset risk field.

    If \`top\` is supplied, the highest-ranked locations are marked and labelled

    directly on the map.

    """

    finite = np.isfinite(dem)

    if not finite.any():

        raise ValueError("DEM contains no finite values")

    filled = np.where(finite, dem, np.nanmin(dem))

    ls = LightSource(azdeg=315, altdeg=45)

    shade = ls.hillshade(filled, vert_exag=1.5, dx=CELL_SIZE_M, dy=CELL_SIZE_M)

    fig, ax = plt.subplots(figsize=(11, 11))

    ax.imshow(shade, cmap="gray", vmin=0, vmax=2.0)

    ax.contour(
        in_canton.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.2,
        alpha=0.8,
    )

    shown = risk > 0

    if shown.any():

        positive = risk[shown]

        im = ax.imshow(
            np.ma.masked_where(~shown, risk),
            cmap="YlOrRd",
            norm=LogNorm(
                vmin=max(float(positive.min()), 1e-3), vmax=float(positive.max())
            ),
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7)

        cbar.set_label("allocated expected damage [CHF/year per 25 m cell, log scale]")
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(format_chf_thousands))

    # Mark highest-ranked locations directly on the map.

    if top is not None and not top.empty:

        marked = top.head(MAP_TOP_N)

        for _, r in marked.iterrows():

            # Convert LV95 coordinates back to raster row/column for imshow axes.

            rr, cc = rasterio.transform.rowcol(transform, r["x_lv95"], r["y_lv95"])

            asset_key = str(r.get("asset", ""))

            abbr = ASSETS.get(asset_key, {}).get(
                "abbr", str(r.get("asset_type", ""))[:3]
            )

            label = f"#{int(r['rank'])} {abbr}"

            ax.scatter(
                cc,
                rr,
                s=55,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=6,
            )

            ax.text(
                cc + 4,
                rr - 4,
                label,
                fontsize=6.5,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    alpha=0.8,
                    linewidth=0.5,
                ),
                zorder=7,
            )

    ax.set_xticks([])

    ax.set_yticks([])

    legend_handles = [Patch(facecolor="black", label="canton boundary")]

    if top is not None and not top.empty and "asset" in top.columns:

        seen_assets = []

        for asset_key in top.head(MAP_TOP_N)["asset"].astype(str):

            if asset_key not in seen_assets:

                seen_assets.append(asset_key)

        for asset_key in seen_assets:

            cfg = ASSETS.get(asset_key, {})

            abbr = cfg.get("abbr", asset_key[:3].title())

            display = cfg.get("display", asset_key.title())

            legend_handles.append(
                Patch(facecolor="none", edgecolor="none", label=f"{abbr} = {display}")
            )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.9,
        fontsize=7,
        handlelength=1.4,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.3,
    )

    if title_label:

        title = (
            f"Spatial EAD:  {title_label} (canton {CANTON_LABEL})\n"
            f"allocated EAD: CHF {format_chf_thousands(risk.sum())}/year"
        )

    else:

        title = (
            f"Spatial EAD ranking (canton {CANTON_LABEL})\n"
            f"total EAD spatially allocated: CHF {format_chf_thousands(risk.sum())}/year"
        )

    ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()

    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")

    plt.close(fig)

    print(f"-> {outpath}")


def plot_ranking(top, outpath, n=12):
    """Bar chart of the highest-ranked contiguous spatial clusters."""

    if top.empty:

        return

    d = top.head(n).iloc[::-1]

    labels = [
        f"#{int(r['rank'])}  {r['asset_type']}  ({r['x_lv95']:.0f}, {r['y_lv95']:.0f})"
        for _, r in d.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 2.2))

    ax.barh(labels, d["risk_chf_per_year"], alpha=0.9)

    for i, value in enumerate(d["risk_chf_per_year"]):

        ax.text(
            value, i, f"  CHF {format_chf_thousands(value)}/yr", va="center", fontsize=9
        )

    ax.set_xlabel("allocated expected damage [CHF/year]")
    ax.xaxis.set_major_formatter(FuncFormatter(format_chf_thousands))

    ax.grid(axis="x", alpha=0.3)

    ax.set_axisbelow(True)

    ax.set_xlim(0, d["risk_chf_per_year"].max() * 1.35)

    ax.set_title(
        f"Highest-ranked locations (canton {CANTON_LABEL})",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")

    plt.close(fig)

    print(f"-> {outpath}")


# --------------------------------------------------------------------- main


def main():

    lines = []

    def log(message=""):

        print(message)

        lines.append(message)

    counts, dem, in_canton, transform, profile, win = load_grid()

    ead = load_ead()

    log("=" * 72)

    log(f"SPATIAL EAD RANKING (canton {CANTON_LABEL})")

    log("=" * 72)

    log(f"  cell_counts : {CELL_COUNTS_TIF}")

    log(f"  EAD source  : {EAD_CSV}")

    log("")

    log("  The EAD values below are read from the final baseline EAD table.")

    log("  This script only distributes those totals spatially using cell_counts.")

    log("")

    risk_total = np.zeros(counts.shape, dtype=np.float32)

    rows = []

    all_rankings = []

    asset_fields = {}

    for asset, cfg in ASSETS.items():

        if asset not in ead.index:

            log(f"  [skip] {asset}: not present in EAD table")

            continue

        mask = load_asset(cfg["raster"], win, counts.shape, asset)

        if mask is None:

            continue

        asset_ead = float(ead.loc[asset, f"{EAD_MEAN_COLNAME}"])

        field, total_weight = risk_for_asset(
            counts=counts,
            asset_mask=mask,
            in_canton=in_canton,
            asset_ead=asset_ead,
        )

        n_asset_cells = int((mask & in_canton).sum())

        n_reached_cells = int((mask & in_canton & (counts > 0)).sum())

        allocated = float(field.sum())

        allocation_ratio = allocated / asset_ead if asset_ead > 0 else np.nan

        # If no mapped asset cell is reached, EAD cannot be spatially allocated.

        # Do not force it onto unrelated cells.

        if total_weight > 0:

            risk_total += field

            asset_fields[asset] = field

            # Rank this asset separately so each ranked location has an explicit

            # asset type. This avoids merging adjacent road/building/etc. cells

            # into one mixed cluster in the final ranking.

            asset_top = rank_segments(field, transform, top_n=None)

            if not asset_top.empty:

                asset_top["asset"] = asset

                asset_top["asset_type"] = cfg.get("display", asset.title())

                all_rankings.append(asset_top)

        row = {
            "asset": asset,
            "ead_input_chf_per_year": round(asset_ead, 2),
            "risk_raster_sum_chf_per_year": round(allocated, 2),
            "allocation_ratio": (
                round(allocation_ratio, 6) if np.isfinite(allocation_ratio) else np.nan
            ),
            "asset_cells_in_canton": n_asset_cells,
            "reached_asset_cells": n_reached_cells,
            "cell_counts_weight_sum": round(total_weight, 2),
        }

        for col in ["events", "events_per_year", "cost_mean"]:

            if col in ead.columns:

                row[col] = ead.loc[asset, col]

        rows.append(row)

        if total_weight > 0:

            log(
                f"  {asset:>10}: input EAD CHF {asset_ead:>10,.0f}/yr -> "
                f"allocated CHF {allocated:>10,.0f}/yr over {n_reached_cells:,} reached cells"
            )

        else:

            log(
                f"  {asset:>10}: input EAD CHF {asset_ead:>10,.0f}/yr -> "
                "NOT ALLOCATED (no asset cell has positive cell_counts)"
            )

    risk_by_asset = pd.DataFrame(rows)

    risk_by_asset.to_csv(f"{OUTDIR}/risk_by_asset.csv", index=False)

    allocatable = (
        risk_by_asset["cell_counts_weight_sum"] > 0
        if not risk_by_asset.empty
        else pd.Series(dtype=bool)
    )

    expected_mapped_total = (
        float(risk_by_asset.loc[allocatable, "ead_input_chf_per_year"].sum())
        if not risk_by_asset.empty
        else 0.0
    )

    raster_total = float(risk_total.sum())

    log("")

    log(f"  allocatable mapped-asset EAD : CHF {expected_mapped_total:,.0f}/yr")

    log(f"  sum of spatial risk raster   : CHF {raster_total:,.0f}/yr")

    if expected_mapped_total > 0:

        drift = abs(raster_total - expected_mapped_total) / expected_mapped_total

        log(
            f"  conservation check           : "
            f"{'OK' if drift < 0.01 else f'DRIFT {drift:.1%} - investigate'}"
        )

    # Write summed raster.

    out_profile = profile.copy()

    out_profile.update(dtype="float32", nodata=0.0, count=1, compress="lzw")

    tif = f"{OUTDIR}/risk_chf_per_year.tif"

    with rasterio.open(tif, "w", **out_profile) as dst:

        dst.write(risk_total, 1)

    log(f"\n-> {tif}")

    # Write one raster and one PNG per asset. These make it easy to inspect

    # exactly which asset type is creating spatial risk at a location.

    for asset, field in asset_fields.items():

        slug = asset.replace(" ", "_")

        display = ASSETS[asset].get("display", asset.title())

        asset_top_map = rank_segments(field, transform, top_n=MAP_TOP_N)

        if not asset_top_map.empty:

            asset_top_map["asset_type"] = display

        plot_risk_map(
            field,
            dem,
            in_canton,
            transform,
            f"{OUTDIR}/risk_{slug}.png",
            title_label=display,
            top=asset_top_map,
        )

    # Global ranking across asset-specific clusters. Because clustering is done

    # separately by asset, every row has an unambiguous asset type.

    if all_rankings:

        top_all = pd.concat(all_rankings, ignore_index=True)

        top_all = top_all.sort_values("risk_chf_per_year", ascending=False).reset_index(
            drop=True
        )

        top_all["rank"] = np.arange(1, len(top_all) + 1)

        # Write the top 50 ranked locations as Flow-Py infrastructure targets.

        # The ID raster stores the global rank (1..50). If two asset clusters

        # overlap in the same cell, the higher-ranked target keeps that cell.

        target_top = top_all.head(TOP_RISK_TARGETS).copy()

        target_ids = np.zeros(counts.shape, dtype=np.uint16)

        for _, r in target_top.iterrows():

            asset = r["asset"]

            cluster_id = int(r["cluster_id"])

            labels, _ = ndimage.label(
                asset_fields[asset] > 0, structure=np.ones((3, 3))
            )

            cluster_mask = labels == cluster_id

            target_ids[cluster_mask & (target_ids == 0)] = int(r["rank"])

        target_binary = (target_ids > 0).astype(np.uint8)

        id_profile = profile.copy()

        id_profile.update(dtype="uint16", nodata=0, count=1, compress="lzw")

        target_id_tif = f"{OUTDIR}/top_risk_targets_id.tif"

        with rasterio.open(target_id_tif, "w", **id_profile) as dst:

            dst.write(target_ids, 1)

        log(f"-> {target_id_tif}")

        binary_profile = profile.copy()

        binary_profile.update(dtype="uint8", nodata=0, count=1, compress="lzw")

        target_binary_tif = f"{OUTDIR}/top_risk_targets_binary.tif"

        with rasterio.open(target_binary_tif, "w", **binary_profile) as dst:

            dst.write(target_binary, 1)

        log(f"-> {target_binary_tif}")

        target_cols = [
            "rank",
            "asset_type",
            "asset",
            "risk_chf_per_year",
            "cells",
            "approx_cell_extent_m",
            "x_lv95",
            "y_lv95",
        ]

        target_top[target_cols].to_csv(f"{OUTDIR}/top_risk_targets.csv", index=False)

        log(f"-> {OUTDIR}/top_risk_targets.csv")

        top = top_all.head(TOP_N_SEGMENTS).copy()

        cols = [
            "rank",
            "asset_type",
            "asset",
            "risk_chf_per_year",
            "cells",
            "approx_cell_extent_m",
            "x_lv95",
            "y_lv95",
        ]

        top = top[cols]

        top.to_csv(f"{OUTDIR}/top_segments.csv", index=False)

        log("")

        log("  TOP 5 LOCATIONS")

        for _, r in top.head(5).iterrows():

            log(
                f"    #{int(r['rank'])}  {r['asset_type']:<12} "
                f"CHF {r['risk_chf_per_year']:>9,.0f}/yr  "
                f"at ({r['x_lv95']:.0f}, {r['y_lv95']:.0f})  "
                f"{int(r['cells'])} cells"
            )

    else:

        top = pd.DataFrame()

    plot_risk_map(
        risk_total, dem, in_canton, transform, f"{OUTDIR}/risk_map.png", top=top
    )

    plot_ranking(top, f"{OUTDIR}/risk_ranking.png")

    log("")

    log("INTERPRETATION")

    log("  - Baseline EAD is empirical and comes directly from ead_by_asset.csv.")

    log("  - The selected Flow-Py map controls only the spatial allocation/ranking.")

    log("  - cell_counts is used as a relative spatial weight, not a probability.")

    log(
        "  - A map with more convergence receives more of an asset's fixed EAD locally."
    )

    log("  - Swapping bare/root/MICP maps does NOT by itself change total EAD.")

    log("  - Therefore an MICP map here is a ranking scenario, not avoided-damage EAD.")

    log("")

    log("CAVEATS")

    log(
        "  - The method assumes relative cell_counts is proportional to relative hit frequency."
    )

    log(
        "  - Only assets listed in ASSETS and having valid rasters are spatially ranked."
    )

    log(
        "  - Ranking clusters are built separately by asset, so every ranked location has an explicit asset type."
    )

    log("  - Separate PNG/TIF maps are also written for each allocatable asset.")

    log(
        f"  - The highest {MAP_TOP_N} ranked locations are marked directly on the PNG maps."
    )

    report = f"{OUTDIR}/risk_report.txt"

    with open(report, "w", encoding="utf-8") as fh:

        fh.write("\n".join(lines) + "\n")

    print(f"\n-> {report}")


if __name__ == "__main__":

    main()
