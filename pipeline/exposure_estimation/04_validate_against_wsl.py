"""
04_validate_against_wsl.py - Does the modelled runout reach where damage
actually happened?

Outputs:
    wsl_hit_rate.csv          hit rate vs canton/road/building controls
    wsl_events_distance.csv   per-event distance to the nearest corridor
    wsl_elevation_check.csv   are corridors stopping ABOVE the damage?
    validation_map.png        damage points over the modelled corridors
    validation_hit_rate.png   hit-rate curve with lift annotated
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
from scipy.ndimage import distance_transform_edt
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
argparse.add_argument(
    "--canton_ids",
    default=[15],
    type=lambda s: [int(x) for x in s.split(",")],
)
args = argparse.parse_args()

# ============================================================== CONFIG =====

FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"
OUTDIR = FLOWPY_DIR + "/validation"
os.makedirs(OUTDIR, exist_ok=True)

ROADS_TIF = "data/infra/roads_binary_25m.tif"
BUILDINGS_TIF = "data/infra/buildings_id_25m.tif"
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
TARGET_CANTON_IDS = args.canton_ids

CELL_SIZE_M = 25.0
RADII_M = [0, 100, 250, 500, 1000]

COL_X = "x-coordinate"
COL_Y = "y-coordinate"
COL_DAMAGE = (
    "extent of damage, low [0.01-0.4], medium [0.4-2], "
    "large/catastrophic [>2] or death [mio. CHF]"
)
COL_DATE = "date"
COL_MUNI = "name of municipality"

PRECISE_ONLY = True  # drop events flagged "Gemeindegebiet"
N_CONTROLS = 2000
SEED = 42
ELEV_SEARCH_M = 1000

# =========================================================== END CONFIG ====


def crop_window():
    meta = json.load(open(f"{INPUT_DIR}/meta.json"))
    r0, r1, c0, c1 = meta["crop_window"]
    return Window(c0, r0, c1 - c0, r1 - r0)


def load_grid():
    """Footprint, canton mask, control layers, DEM, transform - all on one grid."""
    with rasterio.open(f"{FLOWPY_DIR}/z_delta.tif") as src:
        z = src.read(1).astype(np.float32)
        transform = src.transform
    footprint = np.isfinite(z) & (z > 0)

    with rasterio.open(f"{INPUT_DIR}/dem.tif") as src:
        dem = src.read(1).astype(np.float32)
    dem = np.where(dem <= -9000, np.nan, dem)

    win = crop_window()
    with rasterio.open(CANTON_ID_TIF) as src:
        canton = src.read(1, window=win)
    in_canton = np.isin(canton, TARGET_CANTON_IDS)

    controls = {"canton": in_canton.copy()}

    if os.path.exists(ROADS_TIF):
        with rasterio.open(ROADS_TIF) as src:
            controls["roads"] = (src.read(1, window=win) > 0) & in_canton
    else:
        print(f"  [warn] roads raster missing: {ROADS_TIF}")

    if os.path.exists(BUILDINGS_TIF):
        with rasterio.open(BUILDINGS_TIF) as src:
            controls["buildings"] = (src.read(1, window=win) > 0) & in_canton
    else:
        print(f"  [warn] buildings raster missing: {BUILDINGS_TIF}")

    return footprint, in_canton, controls, dem, transform


def sample(arr, transform, xs, ys, default=np.nan):
    """Raster value at LV95 coordinates. `default` where off-grid."""
    h, w = arr.shape
    out = []
    for x, y in zip(xs, ys):
        row, col = rasterio.transform.rowcol(transform, x, y)
        ok = 0 <= row < h and 0 <= col < w
        out.append(arr[row, col] if ok else default)
    return np.array(out)


def load_events(in_canton, transform):
    """WSL landslide-damage events inside the target canton.

    Membership is decided by sampling the canton raster at each event's own
    coordinates - not by the municipality name, which has changed over the
    record and does not cover the 2 km margin around the canton either.
    """
    df = dl.load_wsl_inventory()

    # --- the fix: keep only events whose coordinates fall in the canton ---
    before = len(df)
    inside = sample(
        in_canton.astype(np.uint8), transform, df["x"], df["y"], default=0
    ).astype(bool)
    df = df[inside].reset_index(drop=True)
    print(
        f"  {len(df):,}/{before:,} fall inside canton {TARGET_CANTON_IDS} "
        f"by coordinate"
    )

    if COL_MUNI in df.columns:
        print(f"  municipalities represented: {df[COL_MUNI].nunique()}")

    return df


def random_points(mask, transform, n, seed=SEED):
    """Random LV95 coordinates from the cells of a boolean mask."""
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.array([]), np.array([])
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    xs, ys = rasterio.transform.xy(transform, rows[pick], cols[pick])
    return np.asarray(xs), np.asarray(ys)


# ------------------------------------------------------- elevation diagnostic


def elevation_check(events, footprint, dem, transform, search_m=ELEV_SEARCH_M):
    """Are corridors stopping ABOVE the damage, or nowhere near it?

    If the nearest corridor cells sit consistently HIGHER than the damage
    point, the runout stops too early on the valley floor and a smaller alpha
    is worth testing. If they sit at a similar height but far away, the
    release location is the problem, not the runout length.
    """
    rad = int(search_m / CELL_SIZE_M)
    h, w = footprint.shape
    rows = []

    for _, ev in events.iterrows():
        r, c = rasterio.transform.rowcol(transform, ev["x"], ev["y"])
        if not (0 <= r < h and 0 <= c < w):
            continue
        r0, r1 = max(r - rad, 0), min(r + rad + 1, h)
        c0, c1 = max(c - rad, 0), min(c + rad + 1, w)

        sub_fp, sub_dem = footprint[r0:r1, c0:c1], dem[r0:r1, c0:c1]
        if not sub_fp.any():
            continue

        z_dmg = dem[r, c]
        z_corr = sub_dem[sub_fp & np.isfinite(sub_dem)]
        if z_corr.size == 0 or not np.isfinite(z_dmg):
            continue

        rows.append(
            {
                "x": ev["x"],
                "y": ev["y"],
                "damage_elev_m": round(float(z_dmg), 1),
                "corridor_min_elev_m": round(float(z_corr.min()), 1),
                "corridor_lowest_above_damage_m": round(float(z_corr.min() - z_dmg), 1),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [skip] elevation check: no corridors within search radius")
        return df

    above = df["corridor_lowest_above_damage_m"]
    print(
        f"\n  Elevation check ({len(df)} events with a corridor within "
        f"{search_m:.0f} m):"
    )
    print(
        f"    lowest corridor cell sits {above.mean():+.0f} m relative to "
        f"the damage point, on average"
    )
    print(
        f"    corridor stops ABOVE the damage in "
        f"{100 * (above > 0).mean():.0f}% of cases"
    )
    if (above > 0).mean() > 0.6:
        print("    -> runout stops short. Worth testing a smaller alpha (e.g. 20).")
    else:
        print("    -> runout reaches the right elevation band; the gap is")
        print("       horizontal, so it is a release-location issue.")
    return df


# ------------------------------------------------------------------- figures


def plot_validation_map(footprint, in_canton, dem, events, d_event, transform, outpath):
    """Damage points over the modelled corridors, coloured by distance."""
    filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(filled, vert_exag=1.5, dx=CELL_SIZE_M, dy=CELL_SIZE_M)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(shade, cmap="gray", vmin=0, vmax=2.0)

    # canton outline, so it is obvious which area was actually modelled
    ax.contour(
        in_canton.astype(float),
        levels=[0.5],
        colors="#000000",
        linewidths=1.2,
        alpha=0.8,
    )

    ax.imshow(
        np.ma.masked_where(~footprint, np.ones(footprint.shape, dtype=np.uint8)),
        cmap=ListedColormap(["#ff6d00"]),
        alpha=0.55,
        interpolation="nearest",
    )

    rows, cols = [], []
    for x, y in zip(events["x"], events["y"]):
        r, c = rasterio.transform.rowcol(transform, x, y)
        rows.append(r)
        cols.append(c)
    rows, cols = np.array(rows), np.array(cols)

    bands = [
        (d_event <= 0, "#00c853", "inside a corridor"),
        ((d_event > 0) & (d_event <= 250), "#ffd600", "within 250 m"),
        (d_event > 250, "#d50000", "further than 250 m"),
    ]

    scatter_handles = []
    for m, colour, label in bands:
        if m.any():
            h = ax.scatter(
                cols[m],
                rows[m],
                s=45,
                c=colour,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
                label=f"{label} ({m.sum()})",
            )
            scatter_handles.append(h)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            Patch(facecolor="#ff6d00", alpha=0.55, label="modelled corridors"),
            plt.Line2D([0], [0], color="black", lw=1.2, label="canton boundary"),
        ]
        + scatter_handles,
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )
    ax.set_title(
        f"Recorded landslide damage vs modelled runout:\n"
        f"{100 * (d_event <= 0).mean():.0f}% of damage points inside a corridor "
        f"(n={len(d_event)})",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_hit_rate_curve(rows, outpath):
    """Hit rate vs buffer radius for damage points and each control population."""
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 6))

    damage = (
        df[["radius_m", "damage_hit_rate"]]
        .drop_duplicates(subset=["radius_m"])
        .sort_values("radius_m")
    )
    ax.plot(
        damage["radius_m"],
        100 * damage["damage_hit_rate"],
        "o-",
        lw=2.4,
        label="recorded damage sites",
    )

    markers = {"canton": "^", "roads": "s", "buildings": "D"}
    labels = {
        "canton": "random canton locations",
        "roads": "random road locations",
        "buildings": "random building locations",
    }

    for control_name, group in df.groupby("control_type", sort=False):
        group = group.sort_values("radius_m")
        ax.plot(
            group["radius_m"],
            100 * group["control_hit_rate"],
            marker=markers.get(control_name, "x"),
            linestyle="--",
            lw=1.8,
            label=labels.get(control_name, f"random {control_name} locations"),
        )

    ax.set_xlabel("Maximum distance to a modelled corridor [m]")
    ax.set_ylabel("Locations inside a corridor [%]")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(
        "Do modelled corridors reach recorded damage sites?",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# --------------------------------------------------------------------- main


def main():
    footprint, in_canton, controls, dem, transform = load_grid()

    base_all = 100 * (footprint & in_canton).sum() / in_canton.sum()
    print(f"  {base_all:.1f}% of the canton lies inside a corridor")

    for name, mask in controls.items():
        if not mask.any():
            print(f"  [warn] control '{name}' has no cells inside the canton")
            continue
        base = 100 * (footprint & mask).sum() / mask.sum()
        print(f"  {base:.1f}% of {name.upper()} control cells lie inside a corridor")

    events = load_events(in_canton, transform)

    dist = distance_transform_edt(~footprint, sampling=CELL_SIZE_M)
    d_event = sample(dist, transform, events["x"], events["y"])
    keep = np.isfinite(d_event)
    events, d_event = events[keep].reset_index(drop=True), d_event[keep]

    control_distances = {}
    for i, (name, mask) in enumerate(controls.items()):
        if not mask.any():
            continue
        # Different deterministic seed per control population.
        cx, cy = random_points(mask, transform, N_CONTROLS, seed=SEED + i)
        d = sample(dist, transform, cx, cy)
        d = d[np.isfinite(d)]
        if d.size:
            control_distances[name] = d

    print(f"\n  {len(d_event)} damage points")
    for name, d in control_distances.items():
        print(f"  {len(d):,} random {name} controls")

    rows = []
    for name, d_control in control_distances.items():
        print(f"\n  Control: {name}")
        print(f"  {'radius':>8} {'damage':>9} {'control':>9} {'lift':>7}")

        for r in RADII_M:
            hit_e = float(np.mean(d_event <= r))
            hit_c = float(np.mean(d_control <= r))
            lift = hit_e / hit_c if hit_c > 0 else np.nan
            rows.append(
                {
                    "control_type": name,
                    "radius_m": r,
                    "damage_hit_rate": round(hit_e, 3),
                    "control_hit_rate": round(hit_c, 3),
                    "lift": round(lift, 2) if np.isfinite(lift) else np.nan,
                    "n_damage": len(d_event),
                    "n_control": len(d_control),
                }
            )
            lift_txt = f"{lift:6.2f}x" if np.isfinite(lift) else "   nan "
            print(f"  {r:8d} {hit_e:8.1%} {hit_c:8.1%} {lift_txt}")

        print(
            f"  median distance to nearest corridor: damage {np.median(d_event):.0f} m, "
            f"{name} control {np.median(d_control):.0f} m"
        )

    pd.DataFrame(rows).to_csv(f"{OUTDIR}/wsl_hit_rate.csv", index=False)

    detail = events.copy()
    detail["dist_to_corridor_m"] = d_event
    cols = [
        c for c in [COL_DATE, COL_MUNI, COL_X, COL_Y, COL_DAMAGE] if c in detail.columns
    ]
    detail[cols + ["x", "y", "dist_to_corridor_m"]].to_csv(
        f"{OUTDIR}/wsl_events_distance.csv", index=False
    )

    elev = elevation_check(events, footprint, dem, transform)
    if not elev.empty:
        elev.to_csv(f"{OUTDIR}/wsl_elevation_check.csv", index=False)

    plot_validation_map(
        footprint,
        in_canton,
        dem,
        events,
        d_event,
        transform,
        f"{OUTDIR}/validation_map.png",
    )
    if rows:
        plot_hit_rate_curve(rows, f"{OUTDIR}/validation_hit_rate.png")

    print(f"\n-> {OUTDIR}/")


if __name__ == "__main__":
    main()
