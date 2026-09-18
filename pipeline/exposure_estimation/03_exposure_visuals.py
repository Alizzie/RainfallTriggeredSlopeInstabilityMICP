"""
02_exposure_visuals.py - Figures for the Flow-Py exposure results.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource, LogNorm
from matplotlib.patches import Patch

# ============================================================ Parser =====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)
args = parser.parse_args()

# ============================================================== CONFIG =====
FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"
OUTDIR = FLOWPY_DIR + "/exposure_summary"
os.makedirs(OUTDIR, exist_ok=True)

CELL_SIZE_M = 25.0

LAYERS = {
    "forest": "data/swissTLM3D/tlm_forest_25m_ch.tif",
    "roads": "data/infra/roads_binary_25m.tif",
    "railway": "data/infra/railway_binary_25m.tif",
    "buildings": "data/infra/buildings_id_25m.tif",
    "protection forest": "data/infra/schutzwald_rutschungen_25m.tif",
    "agricultural land": "data/infra/nutzungsgebiete.tif",
    "protected areas": "data/infra/protected_area_25m.tif",
}

MAP1 = [
    "forest",
    "roads",
    "railway",
    "buildings",
    "protection forest",
]

MAP2 = [
    "protected areas",
    "agricultural land",
]


COLORS = {
    "forest": "#00c853",
    "roads": "#ff6d00",
    "railway": "#aa00ff",
    "buildings": "#ff1744",
    "protection forest": "#1c5132",
    "agricultural land": "#B5900E",
    "protected areas": "#E4FB1B",
}

COLORS_SPARED = {
    "forest": "#7d9b82",
    "roads": "#a8968c",
    "railway": "#9b93a8",
    "buildings": "#b09595",
    "protection forest": "#a0c0a088",
    "agricultural land": "#d0c0a09b",
    "protected areas": "#f0f0a088",
}

AFFECTED_EMPTY_COLOR = "#2979ff"
AFFECTED_ALPHA = 0.95
UNAFFECTED_ALPHA = 0.55

# Terrain is dimmed under the overlays so affected cells pop out.
HILLSHADE_VMAX = 2.0

# --- Schutzwald figure ---
SW_SPARED = "#2e7d32"  # outside all corridors
SW_UPSLOPE = "#ff6d00"  # stable itself, hit by material from above
SW_ITSELF = "#d50000"  # unstable itself

# --- terrain rendering ---
CONTOUR_INTERVAL_M = 100  # spacing between contour lines
CONTOUR_INDEX_EVERY = 5  # every Nth contour is thicker and labelled

# =========================================================== END CONFIG ====


def load_window(path, ref_shape, label):
    """Read a national raster cropped to the Flow-Py window."""
    if not os.path.exists(path):
        print(f"  [skip] {label}: {path} not found")
        return None

    meta = json.load(open(f"{INPUT_DIR}/meta.json"))
    r0, r1, c0, c1 = meta["crop_window"]
    with rasterio.open(path) as src:
        arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))

    if arr.shape != ref_shape:
        raise ValueError(f"{label} shape {arr.shape} != grid {ref_shape}")
    return arr


def load_flowpy():
    """z_delta, cell_counts, DEM, release mask."""
    with rasterio.open(f"{FLOWPY_DIR}/z_delta.tif") as src:
        z = src.read(1).astype(np.float32)
    with rasterio.open(f"{FLOWPY_DIR}/cell_counts.tif") as src:
        counts = src.read(1).astype(np.float32)
    with rasterio.open(f"{INPUT_DIR}/dem.tif") as src:
        dem = src.read(1).astype(np.float32)
    with rasterio.open(f"{INPUT_DIR}/release.tif") as src:
        release = src.read(1) > 0

    dem = np.where(dem <= -9000, np.nan, dem)
    return z, counts, dem, release


def hillshade(dem, cell_size=CELL_SIZE_M):
    """Grey relief background. NaN-safe."""
    filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(filled, vert_exag=1.5, dx=cell_size, dy=cell_size)


def draw_terrain(ax, dem):
    """Dimmed hillshade background, so coloured overlays stand out."""
    ax.imshow(hillshade(dem), cmap="gray", vmin=0, vmax=HILLSHADE_VMAX)


def ha(mask):
    """Hectares covered by a boolean mask."""
    return float(mask.sum()) * CELL_SIZE_M**2 / 10_000


def add_contours(ax, dem, interval=CONTOUR_INTERVAL_M, color="#37474f"):
    """Elevation contours from the DEM. Every Nth line is thicker and labelled.

    Contours don't show flow direction explicitly, but flow always runs
    perpendicular to them from high to low, so the terrain stays readable
    without arrows covering the map.
    """
    dem_f = np.where(np.isfinite(dem), dem, np.nan)
    if not np.isfinite(dem_f).any():
        return

    lo = np.floor(np.nanmin(dem_f) / interval) * interval
    hi = np.ceil(np.nanmax(dem_f) / interval) * interval
    levels = np.arange(lo, hi + interval, interval)
    if len(levels) < 2:
        return

    # thin lines everywhere
    ax.contour(dem_f, levels=levels, colors=color, linewidths=0.35, alpha=0.4, zorder=6)

    # thicker, labelled index contours
    index = levels[::CONTOUR_INDEX_EVERY]
    if len(index) >= 2:
        cs = ax.contour(
            dem_f, levels=index, colors=color, linewidths=0.8, alpha=0.7, zorder=7
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")


def overlay(ax, mask, color, alpha=1.0, zorder=3):
    """Paint one boolean mask in a single flat colour."""
    if not mask.any():
        return
    ax.imshow(
        np.ma.masked_where(~mask, np.ones(mask.shape, dtype=np.uint8)),
        cmap=ListedColormap([color]),
        alpha=alpha,
        interpolation="nearest",
        zorder=zorder,
    )


# ------------------------------------------------------------------ plot 1


def plot_overview(z, counts, dem, release, outpath):
    """Terrain + release cells + footprint shaded by path convergence.

    Colour comes from cell_counts, not z_delta: z_delta saturates at max_z and
    carries almost no variation, while path counts span orders of magnitude and
    show where many independent slopes drain into the same channel.
    """
    footprint = np.isfinite(z) & (z > 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_terrain(ax, dem)

    shown = footprint & (counts >= 1)
    vmax = float(counts[shown].max()) if shown.any() else 2.0
    im = ax.imshow(
        np.ma.masked_where(~shown, counts),
        cmap="YlOrRd",
        alpha=0.90,
        norm=LogNorm(vmin=1, vmax=max(vmax, 2.0)),
        interpolation="nearest",
        zorder=3,
    )

    overlay(ax, release, "#6600ff94", alpha=0.60, zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7)
    cbar.set_label(
        "Number of unstable slopes upstream that can reach this cell (log scale)"
    )

    ax.legend(
        handles=[
            Patch(facecolor="#6600ff94", alpha=0.60, label="release cells (FoS < 1)"),
            Patch(facecolor="#d84315", alpha=0.90, label="runout footprint"),
            Patch(facecolor="#9e9e9e", label="not affected"),
        ],
        loc="upper right",
        fontsize=9,
    )

    area_km2 = footprint.sum() * CELL_SIZE_M**2 / 1e6
    window_km2 = z.size * CELL_SIZE_M**2 / 1e6
    ax.set_title(
        f"Modelled landslide runout: \n{int(release.sum()):,} releases, "
        f"{area_km2:.1f} km2 reachable ({100 * area_km2 / window_km2:.0f}% of window)",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# ------------------------------------------------------------------ plot 2


def plot_asset_damage(z, dem, layers, outpath, map_layers=MAP1):
    """Categorical map: which asset type each affected cell represents.

    Assets outside every corridor are drawn first in muted colours, so the
    reader can see what was spared. Without them, every visible asset looks
    like it was hit.
    """
    footprint = np.isfinite(z) & (z > 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_terrain(ax, dem)

    # --- assets OUTSIDE any corridor: muted ---
    for name in map_layers:
        layer = layers.get(name)
        if layer is None:
            continue
        overlay(
            ax,
            (layer > 0) & ~footprint,
            COLORS_SPARED[name],
            alpha=UNAFFECTED_ALPHA,
            zorder=3,
        )

    # --- cells INSIDE a corridor, classified by asset type ---
    cat = np.zeros(z.shape, dtype=np.uint8)
    cat[footprint] = 1  # affected, nothing tracked here

    present = []
    for i, name in enumerate(map_layers, start=2):
        layer = layers.get(name)
        if layer is None:
            continue
        cat[footprint & (layer > 0)] = i
        present.append((i, name))

    colors = ["#00000000", AFFECTED_EMPTY_COLOR] + [COLORS[n] for _, n in present]
    ax.imshow(
        np.ma.masked_where(cat == 0, cat),
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=len(colors) - 1,
        alpha=AFFECTED_ALPHA,
        interpolation="nearest",
        zorder=4,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    handles = [Patch(facecolor=AFFECTED_EMPTY_COLOR, label="affected, no asset")]
    handles += [Patch(facecolor=COLORS[n], label=f"{n}") for _, n in present]
    # handles += [
    #     Patch(facecolor=COLORS_SPARED[n], alpha=UNAFFECTED_ALPHA, label=f"{n} - spared")
    #     for _, n in present
    # ]
    handles += [Patch(facecolor="#9e9e9e", label="not affected, no asset")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    ax.set_title("Affected area by asset type", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# ------------------------------------------------------------------ plot 3


def plot_schutzwald(z, dem, schutzwald, release, outpath):
    """Landslide-protection forest: hit from upslope vs. unstable itself.

    The split matters for the argument. Orange cells are Schutzwald that is
    stable under the model and still ends up in a runout corridor, because the
    release is on the slope above it - the forest is doing its job and is in
    the path anyway. Red cells are Schutzwald that fails itself.

    Schutzwald is a subset of forest, which is why it gets its own figure
    rather than another class in the asset map.
    """
    footprint = np.isfinite(z) & (z > 0)
    sw = schutzwald > 0

    hit_upslope = footprint & sw & ~release
    hit_itself = footprint & sw & release
    spared = sw & ~footprint

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_terrain(ax, dem)

    # the wider runout corridors, faint, for context
    overlay(ax, footprint & ~sw, AFFECTED_EMPTY_COLOR, alpha=0.20, zorder=3)

    overlay(ax, spared, SW_SPARED, alpha=0.55, zorder=4)
    overlay(ax, hit_upslope, SW_UPSLOPE, alpha=0.95, zorder=5)
    overlay(ax, hit_itself, SW_ITSELF, alpha=0.95, zorder=5)

    # add_contours(ax, dem)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(
        handles=[
            Patch(
                facecolor=SW_SPARED,
                alpha=0.55,
                label=f"Protection forest, outside corridors ({ha(spared):,.0f} ha)",
            ),
            Patch(
                facecolor=SW_UPSLOPE,
                label=f"hit from upslope ({ha(hit_upslope):,.0f} ha)",
            ),
            Patch(
                facecolor=SW_ITSELF,
                label=f"unstable itself ({ha(hit_itself):,.0f} ha)",
            ),
            Patch(
                facecolor=AFFECTED_EMPTY_COLOR,
                alpha=0.20,
                label="runout corridors (other)",
            ),
        ],
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )

    total, affected = ha(sw), ha(footprint & sw)
    pct = 100 * affected / total if total else 0.0
    ax.set_title(
        f"Protection forest in runout corridors\n "
        f"{affected:,.0f} of {total:,.0f} ha ({pct:.0f}%)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")

    if affected > 0:
        print(
            f"     of the {affected:,.0f} ha affected, "
            f"{100 * ha(hit_upslope) / affected:.0f}% is hit from upslope "
            f"(stable itself, reached by material from above)"
        )


# ------------------------------------------------------------------ plot 4


def plot_exposure_bars(summary_csv, outpath):
    """Horizontal bars: % of each asset affected, with absolute values labelled."""
    if not os.path.exists(summary_csv):
        print(f"  [skip] {summary_csv} not found - run 01_flowpy_exposure.py first")
        return

    df = pd.read_csv(summary_csv).dropna(subset=["pct_affected"])
    if df.empty:
        print("  [skip] no rows with a percentage to plot")
        return

    df = df.sort_values("pct_affected")

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(df) + 2))
    bars = ax.barh(df["layer"], df["pct_affected"], color="#d84315", alpha=0.9)

    has_paths = "max_paths" in df.columns
    for bar, (_, row) in zip(bars, df.iterrows()):
        unit = "" if row["unit"] == "count" else f" {row['unit']}"
        label = f"{row['affected']:,.0f}{unit} of {row['total']:,.0f}"
        if has_paths and pd.notna(row.get("max_paths")):
            label += f"   (up to {row['max_paths']:,.0f} paths)"
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("% of asset within modelled runout corridors")
    ax.set_xlim(0, min(105, df["pct_affected"].max() * 1.9))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        "Exposure to modelled landslide runout", fontsize=12, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# --------------------------------------------------------------------- main


def main():
    print(f"  Flow-Py output: {FLOWPY_DIR}")
    z, counts, dem, release = load_flowpy()

    footprint = np.isfinite(z) & (z > 0)
    share = 100 * footprint.sum() / footprint.size
    print(f"  grid {z.shape}, {int(release.sum()):,} release cells")
    print(
        f"  reachable: {footprint.sum() * CELL_SIZE_M**2 / 1e6:.1f} km2 "
        f"({share:.0f}% of window)"
    )

    p = counts[footprint]
    print(
        f"  paths per cell: median {np.median(p):.0f}, "
        f"p90 {np.percentile(p, 90):.0f}, max {p.max():.0f}"
    )

    zmax = float(z[footprint].max())
    print(f"  z_delta max {zmax:.0f} m (implied {np.sqrt(2 * 9.81 * zmax):.0f} m/s)")

    layers = {}
    for name, path in LAYERS.items():
        arr = load_window(path, z.shape, name)
        if arr is not None:
            layers[name] = arr
            print(f"  {name:12s} {int((arr > 0).sum()):,} cells in window")

    plot_overview(z, counts, dem, release, f"{OUTDIR}/overview.png")
    plot_asset_damage(z, dem, layers, f"{OUTDIR}/asset_damage_infra.png", MAP1)
    plot_asset_damage(z, dem, layers, f"{OUTDIR}/asset_damage_area.png", MAP2)
    plot_asset_damage(z, dem, layers, f"{OUTDIR}/asset_damage.png", MAP2 + MAP1)

    sw = layers.get("protection forest")
    if sw is not None:
        plot_schutzwald(z, dem, sw, release, f"{OUTDIR}/schutzwald.png")

    plot_exposure_bars(f"{OUTDIR}/exposure_summary.csv", f"{OUTDIR}/exposure_bars.png")


if __name__ == "__main__":
    main()
