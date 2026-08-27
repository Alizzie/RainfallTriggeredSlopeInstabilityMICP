"""
damage_profile.py - Landslide damage/exposure profile.

Fully parametrized (all inputs at the top under CONFIG) so the identical logic
runs nationally or restricted to one/several cantons, just by changing
TARGET_CANTON_IDS - no other code changes needed.

PIPELINE:
  1. Take fos_root.tif as the unstable-area source, over the FULL national
     extent by default (CANTON_ID_TIF is now an ATTRIBUTION layer, not a hard
     clip mask - see below).
  2. Cluster contiguous unstable (FoS<1) pixels into discrete release zones
     (connected-component labeling) - one release point per zone, not one per
     pixel. This is both a performance necessity (a raw pixel-by-pixel run
     would simulate near-duplicate landslides from every adjacent pixel in
     the same unstable patch) and a more physically sensible choice (a
     contiguous unstable slope is one landslide, not one per 25m cell).
  3. For each release point: determine candidate runout direction(s) from the
     DEM (core.runout_model.flow_directions_at). If ambiguous (>1 candidate),
     simulate every candidate and keep the one causing the WORST damage
     (see WORST_DIRECTION_WEIGHTS below - simple, tunable heuristic).
  4. Build the swath mask (path + width buffer), intersect with roads,
     railway, forest, buildings.
  5. Aggregate: per-point CSV (with which canton(s) each swath touches),
     national totals, PER-CANTON totals/percentages, cumulative swath PNG.

CANTON HANDLING - now optional and attributive, not a hard filter:
  - CANTON_ID_TIF is a single raster covering all of Switzerland: 0 = outside
    CH/unassigned, 1-26 (or whatever your KANTONSNUM values are) = canton ID.
  - TARGET_CANTON_IDS = None (default): process the ENTIRE national extent.
    A list like [8] restricts WHICH RELEASE POINTS get simulated (their
    starting cluster must be inside one of the listed cantons) - but a
    swath's damage can still spill into a neighboring canton, and that spill
    is recorded, not clipped away. This models border cases correctly instead
    of silently truncating them at an administrative line.
  - If CANTON_ID_TIF is missing entirely, the whole grid is treated as one
    unlabeled area: national totals still work, but there is no per-canton
    breakdown (logged clearly, not silently skipped).
  - CANTON_NAMES_CSV (optional): two columns, canton_id,name - if present,
    merged into the per-canton summary for readability. If absent, cantons
    are reported by numeric ID only.

OPEN ASSUMPTIONS - NOT SOLVED HERE, FLAGGED EXPLICITLY:
  - Road CLOSURE DURATION ("wie lange ist das geschlossen"): this script
    reports affected road LENGTH/AREA only. Closure duration would need a
    separate model (e.g. tied to estimated debris volume, which this runout
    model does not compute) - not fabricated here.
  - SWATH_WIDTH is a single global assumption, not derived from landslide
    volume/type. Treat it as a scenario parameter and consider it explicitly
    when reporting results.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, xy as transform_xy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

from exposure_estimation import runout_model as rm
from core import data_loader as dl

# ============================================================== CONFIG =====

FOS_TIF = "output/impacts/01_maps/fos_root.tif"  # unstable-area source
DEM_TIF = "data/swissalti_slope/dem_25m_ch.tif"  # elevation, for flow direction

# Optional attribution layer (NOT a hard clip by default - see docstring).
CANTON_ID_TIF = (
    "data/swissboundaries/canton_id_25m.tif"  # 0 = outside CH, 1-26 = canton
)

TARGET_CANTON_IDS = None

# Binary presence rasters (1 = feature present), all pre-aligned to the SAME
# grid as FOS_TIF/DEM_TIF. See QGIS_prep_guide.md for how to build these from
# the ASV road register, railway lines, and building multipolygons.
ROADS_TIF = "data/infra/roads_binary_25m.tif"
RAILWAY_TIF = "data/infra/railway_binary_25m.tif"
FOREST_TIF = (
    "data/swissTLM3D/tlm_forest_25m_ch.tif"  # reuse existing (nonzero = forest)
)

# Buildings: UNIQUE INTEGER ID per building footprint (0 = no building), NOT
# just binary - this is what lets us count distinct buildings inside a swath
# rather than just counting building-pixels.
BUILDINGS_ID_TIF = "data/infra/buildings_id_25m.tif"

OUTDIR = "output/damage_profile/cantonal"
os.makedirs(OUTDIR, exist_ok=True)

# ---- scenario parameters (tune / sweep these; they are ASSUMPTIONS) ----
MAX_RADIUS_M = 300.0  # farthest a landslide is allowed to travel
SWATH_WIDTH_M = 50.0  # width at the TOE (far end) of the runout
HEAD_WIDTH_M = 15.0  # width at the release point (narrow "scarp")
SWATH_SHAPE = "fan"  # "fan" (widens with distance, default) or "rectangle" (old constant-width behavior)
FLAT_SLOPE_DEG = 5.0  # below this local slope -> "reached the valley floor"
UPHILL_TOLERANCE_M = 0.5  # elevation rise (m) that counts as "climbing again"
MIN_CLUSTER_PIXELS = 4  # ignore unstable clusters smaller than this (noise)
ARROW_VISUAL_SCALE = 3.0  # purely cosmetic: multiplies arrow length in the

CELL_SIZE_M = 25.0  # must match the actual raster resolution in use

# How "worst direction" is scored when a release point is ambiguous (>1
# downhill candidate). Simple weighted sum - tune freely.
WORST_DIRECTION_WEIGHTS = {
    "n_buildings": 1000.0,  # buildings dominate the score
    "building_area_m2": 1.0,
    "road_length_m": 5.0,
    "rail_length_m": 10.0,  # railway weighted higher than roads
    "forest_area_m2": 0.01,
}

# =========================================================== END CONFIG ====


def load_all_rasters():
    """Load every input raster, checked against a common reference shape."""
    with rasterio.open(FOS_TIF) as src:
        fos = src.read(1).astype(np.float32)
        transform = src.transform
        shape = fos.shape

    def load_matching(path, label, dtype=np.float32, required=True):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"{label}: {path} not found.")
            print(f"[warn] {label} not found at {path} - treating as all-zero/absent.")
            return np.zeros(shape, dtype=dtype)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(dtype)
        if arr.shape != shape:
            raise ValueError(f"{label} shape {arr.shape} != FoS grid shape {shape}")
        return arr

    dem = load_matching(DEM_TIF, "DEM")
    canton_id = load_matching(
        CANTON_ID_TIF, "canton ID", dtype=np.int16, required=False
    )
    roads = load_matching(ROADS_TIF, "roads", dtype=np.uint8, required=False) > 0
    railway = load_matching(RAILWAY_TIF, "railway", dtype=np.uint8, required=False) > 0
    forest = load_matching(FOREST_TIF, "forest", dtype=np.int32, required=False) > 0
    buildings_id = load_matching(
        BUILDINGS_ID_TIF, "buildings", dtype=np.int32, required=False
    )

    has_canton_data = os.path.exists(CANTON_ID_TIF)

    return {
        "fos": fos,
        "dem": dem,
        "canton_id": canton_id,
        "roads": roads,
        "railway": railway,
        "forest": forest,
        "buildings_id": buildings_id,
        "transform": transform,
        "shape": shape,
        "has_canton_data": has_canton_data,
    }


def find_release_clusters(
    fos, allowed_mask, fos_threshold=1.0, min_pixels=MIN_CLUSTER_PIXELS
):
    """Connected-component labeling of unstable pixels within allowed_mask.
    One release point per cluster: the pixel with the LOWEST FoS in that
    cluster.

    Returns (release_points, labeled_kept, unstable_mask):
      release_points : list of dicts, as before.
      labeled_kept    : same shape as fos, 0 = not part of any KEPT cluster,
                        else the cluster_id - the actual shape of every
                        cluster that produced a release point (distinct from
                        a synthetic circle drawn at one point).
      unstable_mask   : the raw FoS<1 boolean mask BEFORE any clustering or
                        min_pixels filtering - ground truth "where is it
                        unstable at all", independent of how it got grouped.
    """
    unstable = np.isfinite(fos) & (fos < fos_threshold) & allowed_mask
    labeled, n_clusters = ndimage.label(
        unstable, structure=np.ones((3, 3))
    )  # 8-connectivity

    release_points = []
    labeled_kept = np.zeros_like(labeled)
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        n_px = int(cluster_mask.sum())
        if n_px < min_pixels:
            continue
        labeled_kept[cluster_mask] = cluster_id
        rows, cols = np.nonzero(cluster_mask)
        fos_vals = fos[rows, cols]
        worst_idx = np.argmin(fos_vals)
        release_points.append(
            {
                "cluster_id": cluster_id,
                "row": int(rows[worst_idx]),
                "col": int(cols[worst_idx]),
                "cluster_n_pixels": n_px,
                "min_fos": float(fos_vals[worst_idx]),
            }
        )
    return release_points, labeled_kept, unstable


def score_swath(local_mask, row_offset, col_offset, data):
    """Damage metrics for one candidate swath (WINDOWED - local_mask + offset,
    not a full-grid mask): length/area for roads, railway, forest; count +
    area for buildings. Slices the same small window out of each asset layer
    instead of touching the full national arrays."""
    h, w = local_mask.shape
    r0, c0 = row_offset, col_offset
    r1, c1 = r0 + h, c0 + w

    roads_win = data["roads"][r0:r1, c0:c1]
    rail_win = data["railway"][r0:r1, c0:c1]
    forest_win = data["forest"][r0:r1, c0:c1]
    buildings_win = data["buildings_id"][r0:r1, c0:c1]

    roads_len_m = float((local_mask & roads_win).sum()) * CELL_SIZE_M
    rail_len_m = float((local_mask & rail_win).sum()) * CELL_SIZE_M
    forest_area_m2 = float((local_mask & forest_win).sum()) * (CELL_SIZE_M**2)

    building_ids_in_swath = buildings_win[local_mask]
    unique_buildings = np.unique(building_ids_in_swath[building_ids_in_swath > 0])
    n_buildings = len(unique_buildings)
    building_area_m2 = float((building_ids_in_swath > 0).sum()) * (CELL_SIZE_M**2)

    weighted_score = (
        n_buildings * WORST_DIRECTION_WEIGHTS["n_buildings"]
        + building_area_m2 * WORST_DIRECTION_WEIGHTS["building_area_m2"]
        + roads_len_m * WORST_DIRECTION_WEIGHTS["road_length_m"]
        + rail_len_m * WORST_DIRECTION_WEIGHTS["rail_length_m"]
        + forest_area_m2 * WORST_DIRECTION_WEIGHTS["forest_area_m2"]
    )

    return {
        "road_length_m": roads_len_m,
        "rail_length_m": rail_len_m,
        "forest_area_m2": forest_area_m2,
        "n_buildings": n_buildings,
        "building_area_m2": building_area_m2,
        "score": weighted_score,
    }


def cantons_touched(local_mask, row_offset, col_offset, canton_id):
    """Which canton(s) a swath overlaps, and which one has the most pixels -
    windowed version, slices the same small window out of canton_id."""
    h, w = local_mask.shape
    r0, c0 = row_offset, col_offset
    canton_win = canton_id[r0 : r0 + h, c0 : c0 + w]
    ids_in_swath = canton_win[local_mask]
    ids_in_swath = ids_in_swath[ids_in_swath > 0]
    if len(ids_in_swath) == 0:
        return [], None
    unique, counts = np.unique(ids_in_swath, return_counts=True)
    primary = int(unique[np.argmax(counts)])
    touched = sorted(int(u) for u in unique)
    return touched, primary


def process_release_point(rp, data, log):
    """Direction selection -> runout -> swath -> damage scoring for ONE
    release point. Returns a result dict, or None if the point is a local pit
    (no downhill neighbor at all - can't run a landslide there)."""
    dem, shape = data["dem"], data["shape"]
    row, col = rp["row"], rp["col"]

    candidates = rm.flow_directions_at(dem, row, col)
    if not candidates:
        log(f"  cluster {rp['cluster_id']}: local pit, no downhill neighbor - skipped")
        return None

    best = None
    for direction in candidates:
        path, distance, stop_reason = rm.simulate_runout_path(
            dem,
            row,
            col,
            direction,
            CELL_SIZE_M,
            MAX_RADIUS_M,
            flat_slope_deg=FLAT_SLOPE_DEG,
            uphill_tolerance_m=UPHILL_TOLERANCE_M,
        )
        local_mask, r_off, c_off = rm.build_swath_mask(
            path,
            direction,
            SWATH_WIDTH_M,
            CELL_SIZE_M,
            shape,
            head_width_m=HEAD_WIDTH_M,
            shape_mode=SWATH_SHAPE,
        )
        metrics = score_swath(local_mask, r_off, c_off, data)

        if best is None or metrics["score"] > best["metrics"]["score"]:
            best = {
                "direction": direction,
                "path": path,
                "distance_m": distance,
                "stop_reason": stop_reason,
                "local_mask": local_mask,
                "row_offset": r_off,
                "col_offset": c_off,
                "metrics": metrics,
            }

    touched, primary = cantons_touched(
        best["local_mask"], best["row_offset"], best["col_offset"], data["canton_id"]
    )

    end_row, end_col = best["path"][-1]

    # Real-world coordinates (LV95), not just pixel row/col - essential for
    # any downstream use outside this array (a dashboard, GIS export, etc).
    x, y = transform_xy(data["transform"], row, col)
    end_x, end_y = transform_xy(data["transform"], end_row, end_col)

    return {
        "cluster_id": rp["cluster_id"],
        "row": row,
        "col": col,
        "x": round(x, 1),
        "y": round(y, 1),
        "end_row": end_row,
        "end_col": end_col,
        "end_x": round(end_x, 1),
        "end_y": round(end_y, 1),
        "cluster_n_pixels": rp["cluster_n_pixels"],
        "min_fos": rp["min_fos"],
        "n_candidate_directions": len(candidates),
        "chosen_direction": best["direction"],
        "runout_distance_m": round(best["distance_m"], 1),
        "stop_reason": best["stop_reason"],
        "primary_canton_id": primary,
        "cantons_touched": ",".join(map(str, touched)) if touched else "",
        "crosses_canton_border": len(touched) > 1,
        **best["metrics"],
        "_local_mask": best["local_mask"],  # in-memory only, not written to CSV
        "_row_offset": best["row_offset"],
        "_col_offset": best["col_offset"],
    }


def per_canton_summary(data, cumulative_swath, results, canton_names, log):
    """National totals + one row per canton with stock/affected/pct for
    roads, railway, forest, buildings."""
    if not data["has_canton_data"]:
        log(
            "\n[info] No canton ID raster found - skipping per-canton breakdown "
            "(national totals only). See QGIS_prep_guide.md, section 4."
        )
        return None

    canton_id = data["canton_id"]
    present_ids = sorted(int(c) for c in np.unique(canton_id) if c > 0)

    def pct(n, d):
        return round(100 * n / d, 2) if d > 0 else np.nan

    rows = []
    for cid in present_ids:
        in_canton = canton_id == cid

        total_road_px = int((data["roads"] & in_canton).sum())
        total_rail_px = int((data["railway"] & in_canton).sum())
        total_forest_px = int((data["forest"] & in_canton).sum())
        total_buildings = len(
            np.unique(data["buildings_id"][in_canton & (data["buildings_id"] > 0)])
        )

        aff_road_px = int((cumulative_swath & data["roads"] & in_canton).sum())
        aff_rail_px = int((cumulative_swath & data["railway"] & in_canton).sum())
        aff_forest_px = int((cumulative_swath & data["forest"] & in_canton).sum())
        aff_buildings = len(
            np.unique(
                data["buildings_id"][
                    cumulative_swath & in_canton & (data["buildings_id"] > 0)
                ]
            )
        )

        n_points_touching = sum(
            1 for r in results if str(cid) in r["cantons_touched"].split(",")
        )
        n_points_primary = sum(1 for r in results if r["primary_canton_id"] == cid)

        rows.append(
            {
                "canton_id": cid,
                "canton_name": canton_names.get(cid, ""),
                "n_release_points_primary": n_points_primary,
                "n_release_points_touching": n_points_touching,
                "total_road_length_m": total_road_px * CELL_SIZE_M,
                "affected_road_length_m": aff_road_px * CELL_SIZE_M,
                "road_pct_affected": pct(aff_road_px, total_road_px),
                "total_rail_length_m": total_rail_px * CELL_SIZE_M,
                "affected_rail_length_m": aff_rail_px * CELL_SIZE_M,
                "rail_pct_affected": pct(aff_rail_px, total_rail_px),
                "total_forest_area_ha": total_forest_px * (CELL_SIZE_M**2) / 10_000,
                "affected_forest_area_ha": aff_forest_px * (CELL_SIZE_M**2) / 10_000,
                "forest_pct_affected": pct(aff_forest_px, total_forest_px),
                "total_buildings": total_buildings,
                "affected_buildings": aff_buildings,
                "buildings_pct_affected": pct(aff_buildings, total_buildings),
            }
        )

    return pd.DataFrame(rows).sort_values("affected_buildings", ascending=False)


def build_canton_metric_map(canton_id_raster, canton_df, metric_col):
    """Paint every pixel with its enclosing canton's value for one metric -
    a cheap choropleth without needing canton vector geometries, since
    canton_id_raster IS already the polygon boundary information, rasterized."""
    if canton_df is None or canton_df.empty:
        return None
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


def plot_release_point_map(results, allowed_mask, outpath, log):
    """Scatter of every simulated release point, sized by how big its unstable
    cluster was, colored by how many buildings its worst-case swath affects."""
    if not results:
        log("[skip] no release points to plot")
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    background = np.where(allowed_mask, 1, 0).astype(np.uint8)
    ax.imshow(background, cmap="Greys", vmin=0, vmax=3, alpha=0.35)

    rows_ = np.array([r["row"] for r in results])
    cols_ = np.array([r["col"] for r in results])
    end_rows = np.array([r["end_row"] for r in results])
    end_cols = np.array([r["end_col"] for r in results])
    colors = [r["n_buildings"] for r in results]

    # Direction arrows from release point to runout end. Real runout
    # distances are short relative to the plot (MAX_RADIUS_M / CELL_SIZE_M is
    # often just a handful of pixels), so arrows are visually exaggerated by
    # ARROW_VISUAL_SCALE purely for readability - NOT to true scale. The
    # underlying end_x/end_y in the CSV ARE the true, unscaled coordinates.
    dr = (end_rows - rows_) * ARROW_VISUAL_SCALE
    dc = (end_cols - cols_) * ARROW_VISUAL_SCALE
    ax.quiver(
        cols_,
        rows_,
        dc,
        dr,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="dimgray",
        width=0.002,
        headwidth=4,
        headlength=5,
        alpha=0.6,
        zorder=4,
    )

    # Fixed-size dots at the exact release point - size no longer encodes
    # cluster extent (that was the misleading part). See
    # cluster_footprints.png for actual cluster shapes, unstable_mask.png for
    # the raw FoS<1 ground truth.
    sc = ax.scatter(
        cols_,
        rows_,
        s=22,
        c=colors,
        cmap="YlOrRd",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        vmin=0,
        zorder=5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label("Buildings affected by this release point's worst-case swath")
    ax.set_title(
        f"Simulated Landslide Release Points (n={len(results)})\n"
        f"Arrows show runout direction, exaggerated {ARROW_VISUAL_SCALE:.0f}x for visibility "
        f"(true distances are in damage_by_point.csv)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {outpath}")


def plot_unstable_mask(unstable_mask, allowed_mask, outpath, log):
    """Isolated ground-truth view: every pixel with FoS < 1, BEFORE any
    clustering or min_pixels filtering. Compare this against
    cluster_footprints.png and release_points_map.png to judge whether the
    clustering/dots are representing the raw instability sensibly."""
    fig, ax = plt.subplots(figsize=(11, 9))
    display = np.zeros(unstable_mask.shape, dtype=np.uint8)
    display[allowed_mask] = 1
    display[unstable_mask] = 2
    cmap = ListedColormap(["white", "#f0f0f0", "#b71c1c"])
    ax.imshow(display, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Raw Unstable Pixels (FoS < 1) - {int(unstable_mask.sum()):,} pixels, "
        f"before clustering",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {outpath}")


def plot_cluster_footprints(labeled_kept, allowed_mask, outpath, log):
    """Every KEPT cluster shown as its ACTUAL connected-pixel shape, colored
    by cluster ID (cycled through a categorical palette) - the direct fix for
    'why do big clusters look like they contain smaller ones': this shows the
    real footprint instead of an arbitrarily-sized circle at one point."""
    fig, ax = plt.subplots(figsize=(11, 9))
    background = np.where(allowed_mask, 1, 0).astype(np.uint8)
    ax.imshow(background, cmap="Greys", vmin=0, vmax=3, alpha=0.25)

    # cycle cluster IDs through a categorical palette so adjacent clusters
    # are visually distinguishable (mod 19, skip 0=background in tab20)
    display = np.where(labeled_kept > 0, (labeled_kept % 19) + 1, 0)
    display_masked = np.ma.masked_where(labeled_kept == 0, display)
    ax.imshow(
        display_masked,
        cmap="tab20",
        vmin=0,
        vmax=19,
        interpolation="nearest",
        alpha=0.9,
    )

    n_clusters = len(np.unique(labeled_kept)) - 1  # exclude 0
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Cluster Footprints (n={n_clusters:,} distinct clusters, colors cycle/repeat)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {outpath}")


def plot_damage_type_map(data, cumulative_masks, allowed_mask, outpath, log):
    """Categorical map distinguishing WHAT kind of damage each swath pixel
    represents, not just damage/no-damage. Priority order where categories
    overlap (e.g. a road running through forest): buildings > rail > roads >
    forest > swath-only-nothing-there."""
    shape = data["shape"]
    cat = np.zeros(shape, dtype=np.uint8)  # 0 = outside scope
    cat[allowed_mask] = 1  # 1 = in scope, undamaged

    swath, roads_hit, rail_hit, forest_hit, buildings_hit = cumulative_masks
    cat[swath & allowed_mask] = 2  # 2 = swath, no tracked asset there
    cat[forest_hit] = 3
    cat[roads_hit] = 4
    cat[rail_hit] = 5
    cat[buildings_hit] = 6  # highest priority - drawn last/on top

    colors = ["white", "#e8e8e8", "#c9d6c9", "#2e7d32", "#d84315", "#4a148c", "#c62828"]
    labels = [
        "outside scope",
        "in scope, undamaged",
        "swath (no asset)",
        "forest",
        "road",
        "railway",
        "building",
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cat, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3, label=labels[i])
        for i in range(1, 7)
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=9,
        frameon=True,
    )

    ax.set_title("Damage by Asset Type", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {outpath}")


def plot_canton_choropleth(
    canton_id_raster, canton_df, metric_col, metric_label, outpath, log
):
    """Choropleth of one per-canton metric, built by painting canton_id
    pixels with their canton's value - no vector geometries needed.

    Cantons with no simulated data show as light grey (visibly "not
    simulated"), distinct from white (outside Switzerland / nodata)."""
    metric_map = build_canton_metric_map(canton_id_raster, canton_df, metric_col)
    if metric_map is None:
        log(f"[skip] no canton data for choropleth of {metric_col}")
        return

    # distinguish "inside CH but not simulated" (light grey) from "outside CH" (white)
    inside_ch = canton_id_raster > 0
    display = np.where(inside_ch & np.isnan(metric_map), -1.0, metric_map)

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_under("#dddddd")  # "-1" sentinel -> light grey, not white
    im = ax.imshow(display, cmap=cmap, vmin=0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, extend="min"
    )
    cbar.set_label(f"{metric_label}  (grey = canton not simulated)")
    ax.set_title(f"{metric_label} by Canton", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {outpath}")


def crop_to_canton_bbox(data, target_ids, margin_m, cell_size, log):
    """Crop EVERY loaded array down to the bounding box of the target
    canton(s), padded by margin_m (so a runout starting near the canton edge
    still has room to propagate). This is the difference between holding the
    WHOLE COUNTRY in memory (~8+ GB across all layers) vs. just the region
    actually being processed - for a small canton like Glarus this can be a
    50-100x reduction in array size, and is very likely the real fix if a
    canton-restricted run is still slow after the windowed-swath fix."""
    canton_id = data["canton_id"]
    mask = np.isin(canton_id, target_ids)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(
            f"No pixels found for canton ID(s) {target_ids} - cannot crop."
        )

    margin_px = int(np.ceil(margin_m / cell_size))
    h, w = canton_id.shape
    r0 = max(rows.min() - margin_px, 0)
    r1 = min(rows.max() + margin_px + 1, h)
    c0 = max(cols.min() - margin_px, 0)
    c1 = min(cols.max() + margin_px + 1, w)

    old_cells = h * w
    new_cells = (r1 - r0) * (c1 - c0)
    log(
        f"  Cropping to canton bbox: rows[{r0}:{r1}] cols[{c0}:{c1}]  "
        f"({old_cells:,} -> {new_cells:,} cells, {100*new_cells/old_cells:.2f}% of national grid)"
    )

    cropped = {}
    for key in (
        "fos",
        "dem",
        "canton_id",
        "roads",
        "railway",
        "forest",
        "buildings_id",
    ):
        cropped[key] = data[key][r0:r1, c0:c1].copy()
    cropped["shape"] = cropped["fos"].shape
    # Correctly shift the affine transform to match the crop's new origin -
    # this is what keeps row/col -> real x/y conversion valid post-crop.
    cropped["transform"] = data["transform"] * Affine.translation(c0, r0)
    cropped["has_canton_data"] = data["has_canton_data"]
    return cropped


def main():
    t_start = time.time()

    def elapsed():
        return time.time() - t_start

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("DAMAGE PROFILE")
    log("=" * 70)

    t0 = time.time()
    data = load_all_rasters()
    canton_names = dl.load_canton_names()
    log(
        f"[{elapsed():6.1f}s] Rasters loaded ({time.time()-t0:.1f}s)  |  grid shape: {data['shape']}  ({data['shape'][0]*data['shape'][1]:,} total cells)"
    )

    if TARGET_CANTON_IDS is None:
        log("Scope: WHOLE NATIONAL EXTENT (TARGET_CANTON_IDS = None)")
        allowed_mask = np.ones(data["shape"], dtype=bool)
        full_canton_id_for_plot = data["canton_id"]
    else:
        log(f"Scope: restricted to canton IDs {TARGET_CANTON_IDS}")
        if not data["has_canton_data"]:
            raise FileNotFoundError(
                f"TARGET_CANTON_IDS is set but {CANTON_ID_TIF} was not found - "
                f"cannot restrict without the canton ID raster."
            )

        allowed_mask = np.isin(data["canton_id"], TARGET_CANTON_IDS)
        log(f"  allowed pixels: {int(allowed_mask.sum()):,}")
        if int(allowed_mask.sum()) == 0:
            log(
                f"  [!!] WARNING: 0 pixels match canton ID(s) {TARGET_CANTON_IDS}. "
                f"Check that this ID actually appears in {CANTON_ID_TIF} "
                f"(unique values present: {sorted(int(v) for v in np.unique(data['canton_id']) if v > 0)[:15]}...)"
            )

        # Keep the UNCROPPED national canton outline for the final choropleth
        # plot only (small, ~1 int16 array, not the full 7-array ~8GB set) -
        # otherwise the choropleth can only ever show Glarus + crop-margin
        # slivers, since every other canton's pixels get physically discarded
        # by the crop below. This lets the choropleth show all of
        # Switzerland's outlines with only the simulated canton(s) colored.
        full_canton_id_for_plot = data["canton_id"].copy()

        # Crop everything down to just this canton's bounding box (+margin for
        # runout near the edge) BEFORE any further processing.
        t0 = time.time()
        data = crop_to_canton_bbox(
            data, TARGET_CANTON_IDS, MAX_RADIUS_M + SWATH_WIDTH_M, CELL_SIZE_M, log
        )
        allowed_mask = np.isin(data["canton_id"], TARGET_CANTON_IDS)
        log(
            f"[{elapsed():6.1f}s] Cropped to canton bbox ({time.time()-t0:.1f}s)  |  new shape: {data['shape']}"
        )

    t0 = time.time()
    release_points, labeled_kept, unstable_mask = find_release_clusters(
        data["fos"], allowed_mask
    )
    log(
        f"[{elapsed():6.1f}s] Clustering done ({time.time()-t0:.1f}s)  |  "
        f"unstable clusters found (>= {MIN_CLUSTER_PIXELS} px): {len(release_points)}"
    )

    if len(release_points) > 2000:
        log(
            f"  [note] {len(release_points):,} clusters found. For a small alpine canton "
            f"with widespread steep terrain this can be entirely legitimate (Glarus "
            f"itself produces ~2900) - this is no longer flagged as suspicious by "
            f"default. Only worth double-checking if this number is unexpectedly close "
            f"to the FULL NATIONAL cluster count for the same canton restriction."
        )

    results = []
    cumulative_swath = np.zeros(data["shape"], dtype=bool)
    cumulative_roads_hit = np.zeros(data["shape"], dtype=bool)
    cumulative_rail_hit = np.zeros(data["shape"], dtype=bool)
    cumulative_forest_hit = np.zeros(data["shape"], dtype=bool)
    cumulative_buildings_hit = np.zeros(data["shape"], dtype=bool)

    t_loop_start = time.time()
    progress_every = max(1, len(release_points) // 20)  # ~20 progress lines total

    for i, rp in enumerate(release_points):
        result = process_release_point(rp, data, log)

        if (i + 1) % progress_every == 0 or (i + 1) == len(release_points):
            done = i + 1
            loop_elapsed = time.time() - t_loop_start
            per_point = loop_elapsed / done
            remaining = per_point * (len(release_points) - done)
            log(
                f"  [{elapsed():6.1f}s] {done}/{len(release_points)} release points "
                f"({per_point*1000:.1f} ms/point, ~{remaining:.0f}s remaining)"
            )

        if result is None:
            continue
        local_mask = result.pop("_local_mask")
        r_off = result.pop("_row_offset")
        c_off = result.pop("_col_offset")

        rm.paste_local_mask(cumulative_swath, local_mask, r_off, c_off)
        h, w = local_mask.shape
        r1, c1 = r_off + h, c_off + w
        cumulative_roads_hit[r_off:r1, c_off:c1] |= (
            local_mask & data["roads"][r_off:r1, c_off:c1]
        )
        cumulative_rail_hit[r_off:r1, c_off:c1] |= (
            local_mask & data["railway"][r_off:r1, c_off:c1]
        )
        cumulative_forest_hit[r_off:r1, c_off:c1] |= (
            local_mask & data["forest"][r_off:r1, c_off:c1]
        )
        cumulative_buildings_hit[r_off:r1, c_off:c1] |= local_mask & (
            data["buildings_id"][r_off:r1, c_off:c1] > 0
        )
        results.append(result)

    log(
        f"[{elapsed():6.1f}s] Release-point loop done ({time.time()-t_loop_start:.1f}s total)  |  "
        f"simulated {len(results)} (skipped {len(release_points) - len(results)} local pits)"
    )

    n_crossing = sum(1 for r in results if r["crosses_canton_border"])
    if data["has_canton_data"] and n_crossing:
        log(f"  {n_crossing} release points' damage crosses a canton border")

    df = pd.DataFrame(results)
    per_point_csv = f"{OUTDIR}/damage_by_point.csv"
    df.to_csv(per_point_csv, index=False)
    log(f"-> {per_point_csv}")

    # ---- national totals (deduplicated via the OR'd cumulative mask) ----
    total_road_px = (
        int((data["roads"] & allowed_mask).sum())
        if TARGET_CANTON_IDS
        else int(data["roads"].sum())
    )
    total_rail_px = (
        int((data["railway"] & allowed_mask).sum())
        if TARGET_CANTON_IDS
        else int(data["railway"].sum())
    )
    total_forest_px = (
        int((data["forest"] & allowed_mask).sum())
        if TARGET_CANTON_IDS
        else int(data["forest"].sum())
    )
    total_buildings = (
        len(np.unique(data["buildings_id"][allowed_mask & (data["buildings_id"] > 0)]))
        if TARGET_CANTON_IDS
        else len(np.unique(data["buildings_id"][data["buildings_id"] > 0]))
    )

    affected_road_px = int((cumulative_swath & data["roads"]).sum())
    affected_rail_px = int((cumulative_swath & data["railway"]).sum())
    affected_forest_px = int((cumulative_swath & data["forest"]).sum())
    affected_building_ids = np.unique(
        data["buildings_id"][
            cumulative_swath
            & (allowed_mask if TARGET_CANTON_IDS else True)
            & (data["buildings_id"] > 0)
        ]
    )
    n_affected_buildings = len(affected_building_ids)

    def pct(n, d):
        return round(100 * n / d, 2) if d > 0 else np.nan

    summary = {
        "n_release_points": len(results),
        "total_road_length_m": total_road_px * CELL_SIZE_M,
        "affected_road_length_m": affected_road_px * CELL_SIZE_M,
        "road_pct_affected": pct(affected_road_px, total_road_px),
        "total_rail_length_m": total_rail_px * CELL_SIZE_M,
        "affected_rail_length_m": affected_rail_px * CELL_SIZE_M,
        "rail_pct_affected": pct(affected_rail_px, total_rail_px),
        "total_forest_area_ha": total_forest_px * (CELL_SIZE_M**2) / 10_000,
        "affected_forest_area_ha": affected_forest_px * (CELL_SIZE_M**2) / 10_000,
        "forest_pct_affected": pct(affected_forest_px, total_forest_px),
        "total_buildings": total_buildings,
        "affected_buildings": n_affected_buildings,
        "buildings_pct_affected": pct(n_affected_buildings, total_buildings),
    }

    log(f"\n{'=' * 70}")
    scope_header = (
        "NATIONAL SUMMARY"
        if TARGET_CANTON_IDS is None
        else f"SUMMARY (restricted to canton IDs {TARGET_CANTON_IDS})"
    )
    log(scope_header)
    log(f"{'=' * 70}")
    for k, v in summary.items():
        log(f"  {k:28s}: {v}")

    summary_df = pd.DataFrame([summary])
    summary_csv = f"{OUTDIR}/damage_summary_national.csv"
    summary_df.to_csv(summary_csv, index=False)
    log(f"\n-> {summary_csv}")

    # ---- per-canton breakdown ----
    t0 = time.time()
    canton_df = per_canton_summary(data, cumulative_swath, results, canton_names, log)
    log(f"[{elapsed():6.1f}s] Per-canton summary done ({time.time()-t0:.1f}s)")
    if canton_df is not None:
        canton_csv = f"{OUTDIR}/damage_summary_by_canton.csv"
        canton_df.to_csv(canton_csv, index=False)
        log(f"\n{'=' * 70}")
        log("TOP 10 CANTONS BY AFFECTED BUILDINGS")
        log(f"{'=' * 70}")
        log(canton_df.head(10).to_string(index=False))
        log(f"\n-> {canton_csv}")

    # ---- cumulative swath map ----
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(11, 9))
    display = np.zeros(data["shape"], dtype=np.uint8)
    display[allowed_mask] = 1
    display[cumulative_swath] = 2
    ax.imshow(display, cmap="Greens", vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    scope_label = (
        "National" if TARGET_CANTON_IDS is None else f"Cantons {TARGET_CANTON_IDS}"
    )
    ax.set_title(
        f"{scope_label}: Cumulative Landslide Damage Corridors ({len(results)} release points)",
        fontsize=13,
    )
    fig.tight_layout()
    map_path = f"{OUTDIR}/cumulative_damage_swath.png"
    fig.savefig(map_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"-> {map_path}")

    # ---- NEW: raw unstable mask (ground truth, before clustering) ----
    plot_unstable_mask(unstable_mask, allowed_mask, f"{OUTDIR}/unstable_mask.png", log)

    # ---- NEW: actual cluster footprints (fixes the misleading circle-size issue) ----
    plot_cluster_footprints(
        labeled_kept, allowed_mask, f"{OUTDIR}/cluster_footprints.png", log
    )

    # ---- clustered release-point map (now: fixed dot size + direction arrows) ----
    plot_release_point_map(
        results, allowed_mask, f"{OUTDIR}/release_points_map.png", log
    )

    # ---- NEW: damage-by-asset-type map ----
    plot_damage_type_map(
        data,
        (
            cumulative_swath,
            cumulative_roads_hit,
            cumulative_rail_hit,
            cumulative_forest_hit,
            cumulative_buildings_hit,
        ),
        allowed_mask,
        f"{OUTDIR}/damage_by_asset_type.png",
        log,
    )

    # ---- NEW: canton choropleth ----
    if canton_df is not None:
        plot_canton_choropleth(
            full_canton_id_for_plot,
            canton_df,
            "buildings_pct_affected",
            "% of buildings affected",
            f"{OUTDIR}/canton_choropleth_buildings_pct.png",
            log,
        )
    log(f"[{elapsed():6.1f}s] All plots done ({time.time()-t0:.1f}s)")

    with open(f"{OUTDIR}/damage_profile_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\n-> {OUTDIR}/damage_profile_log.txt")
    print(f"\nTOTAL RUNTIME: {elapsed():.1f}s")


if __name__ == "__main__":
    main()
