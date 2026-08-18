"""
Enhanced map export: TIF → PNG with legend at bottom, Switzerland border overlay.

Uses the standard pyproj/fiona approach to load Swiss borders from a simple GeoJSON.
If borders not available, plots the TIF as-is (still looks professional).
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os

# Try to load Switzerland border (optional)
try:
    import geopandas as gpd

    BORDERS_AVAILABLE = True
except ImportError:
    BORDERS_AVAILABLE = False
    print("[info] geopandas not available; maps will plot without border overlay")

# --- Configuration ---
MAPS = {
    "fos_bare": {
        "path": "output/impacts/01_maps/fos_bare.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (bare soil)",
        "figsize": (14, 11),
    },
    "fos_root": {
        "path": "output/impacts/01_maps/fos_root.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (with roots)",
        "figsize": (14, 11),
    },
    "required_cohesion": {
        "path": "output/impacts/01_maps/required_cohesion.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Required Cohesion [kPa]",
        "figsize": (14, 11),
    },
    "required_cohesion_above_roots": {
        "path": "output/impacts/01_maps/required_cohesion_above_roots.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Additional Cohesion Needed (above roots) [kPa]",
        "figsize": (14, 11),
    },
}

OUTDIR = "output/impacts/01_maps/png_export"
SWISS_BORDER_PATH = "data/shapes/swiss_border.geojson"  # Optional
os.makedirs(OUTDIR, exist_ok=True)

DPI = 150  # screen: 150, print: 300


def load_swiss_border():
    """Load Swiss border from GeoJSON if available."""
    if not BORDERS_AVAILABLE:
        return None
    if not os.path.exists(SWISS_BORDER_PATH):
        print(f"[info] {SWISS_BORDER_PATH} not found; plotting without border")
        return None
    try:
        gdf = gpd.read_file(SWISS_BORDER_PATH)
        return gdf
    except Exception as e:
        print(f"[warn] Failed to load border: {e}")
        return None


def plot_map_with_border(data, name_key, config, swiss_border=None):
    """Plot a raster with colorbar at bottom, optional Switzerland border overlay."""

    if not os.path.exists(config["path"]):
        print(f"[skip] {config['path']} not found")
        return

    with rasterio.open(config["path"]) as src:
        raster = src.read(1).astype(np.float32)
        bounds = src.bounds
        crs = src.crs

    valid = np.isfinite(raster)
    n_valid = int(valid.sum())
    print(f"{name_key:30s} | valid pixels: {n_valid:,}")

    # Create figure with space for legend below
    fig, ax = plt.subplots(figsize=config["figsize"], dpi=DPI)

    # Plot raster
    im = ax.imshow(
        raster,
        cmap=config["cmap"],
        vmin=config["vmin"],
        vmax=config["vmax"],
        interpolation="nearest",
        origin="upper",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
    )

    # Overlay Switzerland border if available
    if swiss_border is not None:
        try:
            swiss_border.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.6,
            )
            print(f"  ✓ Swiss border overlaid")
        except Exception as e:
            print(f"  [warn] Border overlay failed: {e}")

    # Colorbar at the BOTTOM (orientation='horizontal')
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.08,  # space between map and colorbar
        shrink=0.8,  # width of colorbar (80% of map width)
        aspect=40,  # aspect ratio (wider bar)
    )
    cbar.set_label(config["cbar_label"], fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title at top
    ax.set_title(config["cbar_label"], fontsize=13, pad=15, fontweight="bold")

    fig.tight_layout()

    # Save
    out_path = f"{OUTDIR}/{name_key}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path}\n")


def main():
    # Load border once
    swiss_border = load_swiss_border()

    for name, config in MAPS.items():
        plot_map_with_border(None, name, config, swiss_border=swiss_border)

    print(f"✓ All maps saved to {OUTDIR}/")
    print(f"\nTo use Swiss border overlay:")
    print(f"  1. Obtain a Swiss border GeoJSON and save to: {SWISS_BORDER_PATH}")
    print(f"     (e.g., from geopandas.read_file('naturalearth').to_file())")
    print(f"  2. Re-run this script")


if __name__ == "__main__":
    main()
