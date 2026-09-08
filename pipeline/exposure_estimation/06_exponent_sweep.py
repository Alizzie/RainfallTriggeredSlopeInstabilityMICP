"""
07_exponent_sweep.py - How wide are the predicted landslide paths?

Run first Flow-Py with different exponent settings, then run this script to measure the
typical predicted path width and compare with recorded damage.
The exponent controls how much the predicted paths spread sideways. A higher exponent
produces narrower paths, a lower exponent produces wider paths.
"""

import glob
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from core import data_loader as dl

# ==============================================================
# Configuration
# ==============================================================

EXPONENTS = [8, 25, 50, 75, 100, 150, 200]

RUN_DIR_TEMPLATE = "output/flowpy/exp_{exp}"

INPUT_DIR = "output/flowpy/glarus/bare_1/flowpy_input/baseline"
SWEEP_CSV = "output/flowpy/exponent_sweep/exponent_sweep.csv"
OUTDIR = "output/flowpy/exponent_sweep"

CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
TARGET_CANTON_IDS = [15]

CELL_SIZE_M = 25.0
TOE_FRACTION = 0.33
CHOSEN_EXP = 75

# --- accuracy check against recorded damage ---
ROADS_TIF = "data/infra/roads_binary_25m.tif"

COL_PROCESS = "main process"
COL_PRECISION = (
    "centre of damage is known (key: bekannt = known) or not known "
    "(key: Gemeindegebiet = not known)"
)

PRECISE_ONLY = True
N_CONTROLS = 2000
SEED = 42

os.makedirs(OUTDIR, exist_ok=True)


# ==============================================================
# Data loading
# ==============================================================


def load_context():
    """Load the canton mask and elevation model."""

    with open(f"{INPUT_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    r0, r1, c0, c1 = meta["crop_window"]
    win = Window(c0, r0, c1 - c0, r1 - r0)

    with rasterio.open(CANTON_ID_TIF) as src:
        canton = src.read(1, window=win)

    in_canton = np.isin(canton, TARGET_CANTON_IDS)

    with rasterio.open(f"{INPUT_DIR}/dem.tif") as src:
        dem = src.read(1).astype(np.float32)

    dem = np.where(dem <= -9000, np.nan, dem)

    return in_canton, dem


def find_run(exp):
    """Find the most recent model output for one tested setting."""

    pattern = f"{RUN_DIR_TEMPLATE.format(exp=exp)}/**/z_delta.tif"

    hits = glob.glob(pattern, recursive=True)

    return sorted(hits)[-1] if hits else None


# ==============================================================
# Accuracy check against recorded damage
# ==============================================================


def load_grid_geometry():
    """Transform of the model grid, plus the road mask on the same window.

    Kept separate from load_context so the width calculation is untouched.
    """

    with open(f"{INPUT_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    r0, r1, c0, c1 = meta["crop_window"]
    win = Window(c0, r0, c1 - c0, r1 - r0)

    with rasterio.open(CANTON_ID_TIF) as src:
        transform = rasterio.windows.transform(win, src.transform)

    roads = None

    if os.path.exists(ROADS_TIF):
        with rasterio.open(ROADS_TIF) as src:
            roads = src.read(1, window=win) > 0

    return transform, roads


def sample_raster(arr, transform, xs, ys, default=np.nan):
    """Raster value at a list of LV95 coordinates."""

    h, w = arr.shape
    out = []

    for x, y in zip(xs, ys):
        row, col = rasterio.transform.rowcol(transform, x, y)

        inside = 0 <= row < h and 0 <= col < w

        out.append(arr[row, col] if inside else default)

    return np.array(out)


def load_damage_points(in_canton, transform):
    """Recorded landslide damage locations inside the canton.

    Membership comes from the coordinates rather than the municipality name,
    because the model grid extends past the canton border.
    """

    df = dl.load_wsl_inventory()

    df = df[df[COL_PROCESS].astype(str).str.contains("Rutschung", na=False)]

    if PRECISE_ONLY and COL_PRECISION in df.columns:
        df = df[df[COL_PRECISION].astype(str).str.strip() == "bekannt"]

    inside = sample_raster(
        in_canton.astype(np.uint8),
        transform,
        df["x"],
        df["y"],
        default=0,
    ).astype(bool)

    return df[inside].reset_index(drop=True)


def road_control_points(in_canton, roads, transform, n=N_CONTROLS, seed=SEED):
    """Random locations on the road network, for comparison.

    Damage points sit on roads in valleys, so comparing them against random
    canton points would test geography rather than the model. Controls are
    drawn from the same population instead.
    """

    mask = (roads & in_canton) if roads is not None else in_canton

    rows, cols = np.nonzero(mask)

    if len(rows) == 0:
        return np.array([]), np.array([])

    rng = np.random.default_rng(seed)

    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)

    xs, ys = rasterio.transform.xy(transform, rows[pick], cols[pick])

    return np.asarray(xs), np.asarray(ys)


def accuracy_stats(z_path, in_canton, transform, events, cx, cy):
    """Share of recorded damage sites and of control points inside a path."""

    with rasterio.open(z_path) as src:
        z = src.read(1).astype(np.float32)

    path_mask = np.isfinite(z) & (z > 0) & in_canton

    if not path_mask.any():
        return {"hit_0m": np.nan, "control_0m": np.nan}

    hit_event = sample_raster(
        path_mask.astype(np.uint8),
        transform,
        events["x"],
        events["y"],
        default=0,
    )

    hit_control = sample_raster(
        path_mask.astype(np.uint8),
        transform,
        cx,
        cy,
        default=0,
    )

    return {
        "hit_0m": float(np.mean(hit_event > 0)) if hit_event.size else np.nan,
        "control_0m": float(np.mean(hit_control > 0)) if hit_control.size else np.nan,
        "n_damage": int(hit_event.size),
        "n_control": int(hit_control.size),
    }


# ==============================================================
# Width calculation
# ==============================================================


def width_stats(z_path, in_canton, dem):
    """
    Measure the typical predicted path width.

    Width is calculated from the distance between each path cell and the
    nearest cell outside the path.

    The function returns:
        median path width across the canton
        median path width in the bottom third of the elevation range
        total predicted path area
    """

    with rasterio.open(z_path) as src:
        z = src.read(1).astype(np.float32)

    path_mask = np.isfinite(z) & (z > 0) & in_canton

    if not path_mask.any():
        return None

    half_width = distance_transform_edt(
        path_mask,
        sampling=CELL_SIZE_M,
    )

    width = 2.0 * half_width[path_mask]

    elevation = dem[path_mask]
    finite = np.isfinite(elevation)

    bottom_width = np.array([])

    if finite.any():
        low = np.nanmin(elevation)
        high = np.nanmax(elevation)

        bottom_limit = low + TOE_FRACTION * (high - low)

        bottom_width = width[finite & (elevation <= bottom_limit)]

    return {
        "median_width_m": float(np.median(width)),
        "toe_median_width_m": (
            float(np.median(bottom_width)) if bottom_width.size else np.nan
        ),
        "area_km2": (float(path_mask.sum()) * CELL_SIZE_M**2 / 1e6),
    }


# ==============================================================
# Figure
# ==============================================================


def plot(df, outpath):
    """
    Make a minimal two panel figure.

    Left panel:
        compares recorded damage with similar road locations

    Right panel:
        shows how predicted path width changes with the setting
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.0),
    )

    x_label = "Exponent"

    # ==========================================================
    # Left panel
    # ==========================================================

    ax = axes[0]

    if "hit_0m" in df.columns and "control_0m" in df.columns:
        recorded = 100 * df["hit_0m"]
        comparison = 100 * df["control_0m"]

        ax.plot(
            df["exp"],
            recorded,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            color="#C62828",
            label="Recorded damage",
        )

        ax.plot(
            df["exp"],
            comparison,
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color="#78909C",
            label="Similar road locations",
        )

        ymax = max(
            recorded.max(),
            comparison.max(),
        )

        ax.set_ylim(
            0,
            max(60, ymax * 1.22),
        )

        ax.set_ylabel(
            "Locations inside a predicted path (%)",
            fontsize=10,
            color="#333333",
        )

        ax.set_title(
            "Does the model find recorded damage?",
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=10,
        )

        ax.legend(
            frameon=False,
            fontsize=9,
            loc="upper right",
        )

    else:
        ax.text(
            0.5,
            0.5,
            "Damage comparison data not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )

        ax.set_title(
            "Does the model find recorded damage?",
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=10,
        )

    ax.set_xlabel(
        x_label,
        fontsize=9.2,
        color="#444444",
        labelpad=10,
    )

    ax.set_xticks([8, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.grid(
        True,
        color="#ECECEC",
        linewidth=0.8,
    )

    # ==========================================================
    # Right panel
    # ==========================================================

    ax = axes[1]

    ax.plot(
        df["exp"],
        df["median_width_m"],
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color="#1976D2",
        label="Median path width",
    )

    ax.plot(
        df["exp"],
        df["toe_median_width_m"],
        marker="o",
        markersize=5.0,
        linewidth=1.8,
        color="#E64A19",
        label="Median path width near slope bottom",
    )

    ymax = max(
        df["median_width_m"].max(),
        df["toe_median_width_m"].max(),
    )

    ax.set_ylim(
        0,
        max(330, ymax * 1.15),
    )

    ax.set_title(
        "How wide are the predicted paths?",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    ax.set_ylabel(
        "Predicted path width (m)",
        fontsize=10,
        color="#333333",
    )

    ax.set_xlabel(
        x_label,
        fontsize=9.2,
        color="#444444",
        labelpad=10,
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    ax.set_xticks([8, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.grid(
        True,
        color="#ECECEC",
        linewidth=0.8,
    )

    # ==========================================================
    # Figure text
    # ==========================================================

    fig.suptitle(
        "Choosing how much predicted landslide paths should spread sideways",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(
        outpath,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure: {outpath}")


# ==============================================================
# Main
# ==============================================================


def main():
    in_canton, dem = load_context()

    transform, roads = load_grid_geometry()

    events = load_damage_points(in_canton, transform)

    cx, cy = road_control_points(in_canton, roads, transform)

    print()
    print(
        f"Recorded damage points inside the canton: {len(events)}   "
        f"comparison road locations: {len(cx)}"
    )

    rows = []

    print()
    print(
        f"{'setting':>10}"
        f"{'typical width':>16}"
        f"{'bottom width':>16}"
        f"{'path area':>14}"
        f"{'damage found':>15}"
    )

    for exp in EXPONENTS:
        path = find_run(exp)

        if path is None:
            print(f"{exp:10d}" f"{'no run found':>16}")
            continue

        stats = width_stats(
            path,
            in_canton,
            dem,
        )

        if stats is None:
            print(f"{exp:10d}" f"{'empty path':>16}")
            continue

        accuracy = accuracy_stats(
            path,
            in_canton,
            transform,
            events,
            cx,
            cy,
        )

        rows.append(
            {
                "exp": exp,
                **stats,
                **accuracy,
            }
        )

        print(
            f"{exp:10d}"
            f"{stats['median_width_m']:15.0f} m"
            f"{stats['toe_median_width_m']:15.0f} m"
            f"{stats['area_km2']:13.0f} km2"
            f"{100 * accuracy['hit_0m']:14.0f} %"
        )

    if not rows:
        print()
        print("No model runs were found.")
        return

    df = pd.DataFrame(rows)

    if os.path.exists(SWEEP_CSV):
        earlier = pd.read_csv(SWEEP_CSV)

        earlier = earlier.drop(
            columns=[c for c in df.columns if c != "exp" and c in earlier.columns]
        )

        df = df.merge(
            earlier,
            on="exp",
            how="left",
        )

    csv_path = f"{OUTDIR}/corridor_width.csv"
    figure_path = f"{OUTDIR}/corridor_width.png"

    df.to_csv(
        csv_path,
        index=False,
    )

    plot(
        df,
        figure_path,
    )

    print()
    print("Summary")

    if "hit_0m" in df.columns and "control_0m" in df.columns:
        gap = 100 * (df["hit_0m"] - df["control_0m"])

        print("Recorded damage is found more often than similar road locations.")

        print(
            f"The difference stays between "
            f"{gap.min():.0f} and {gap.max():.0f} percentage points "
            f"across the tested settings."
        )

        print(
            "Because this changes little, the damage check alone does not "
            "clearly select one setting."
        )

    print(
        f"Typical predicted path width changes from "
        f"{df['median_width_m'].max():.0f} m to "
        f"{df['median_width_m'].min():.0f} m across the tested values."
    )

    chosen = df[df["exp"] == CHOSEN_EXP]

    if not chosen.empty:
        selected = chosen.iloc[0]

        print()
        print(
            f"At the chosen value ({CHOSEN_EXP}), "
            f"the typical predicted path width is "
            f"{selected['median_width_m']:.0f} m."
        )

        print(
            f"Near the bottom of the slope, the typical width is "
            f"{selected['toe_median_width_m']:.0f} m."
        )

        print(f"The predicted paths cover about " f"{selected['area_km2']:.0f} km2.")

    print()
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
