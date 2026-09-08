"""
02_export_maps.py - Export FoS/cohesion rasters to PNG and create MICP overlays.

Overlay classes:
0 = no MICP  & stable
1 = no MICP  & unstable
2 = MICP yes & unstable
3 = MICP yes & stable
"""

import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

try:
    import geopandas as gpd

    BORDERS_AVAILABLE = True
except ImportError:
    BORDERS_AVAILABLE = False
    print("[info] geopandas not available; maps will plot without border overlay")


M_PP = 0.12

PARENT_DIR = f"output/impacts/01_maps/m_pp_{M_PP}"
OUTDIR = f"{PARENT_DIR}/png_export"

MICP_INJECTION_TIF = "data/injection/injection_map.tif"
SWISS_BORDER_PATH = "data/shapes/swiss_border.geojson"

DPI = 150

OVERLAY_REFERENCES = {
    "bare": "fos_bare",
    "root": "fos_root",
}

os.makedirs(OUTDIR, exist_ok=True)


MAPS = {
    "fos_bare": {
        "path": f"{PARENT_DIR}/fos_bare_{M_PP}.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (bare soil)",
        "figsize": (14, 11),
    },
    "fos_root": {
        "path": f"{PARENT_DIR}/fos_root_{M_PP}.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (roots)",
        "figsize": (14, 11),
    },
    "fos_micp": {
        "path": f"{PARENT_DIR}/fos_micp_{M_PP}.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (MICP)",
        "figsize": (14, 11),
    },
    "fos_root_micp": {
        "path": f"{PARENT_DIR}/fos_root_micp_{M_PP}.tif",
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "cbar_label": "Factor of Safety (roots + MICP)",
        "figsize": (14, 11),
    },
    "required_cohesion": {
        "path": f"{PARENT_DIR}/required_cohesion_{M_PP}.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Required Cohesion [kPa]",
        "figsize": (14, 11),
    },
    "required_cohesion_above_roots": {
        "path": f"{PARENT_DIR}/required_cohesion_above_roots_{M_PP}.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Additional Cohesion Needed above Roots [kPa]",
        "figsize": (14, 11),
    },
    "required_cohesion_above_micp": {
        "path": f"{PARENT_DIR}/required_cohesion_above_micp_{M_PP}.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Additional Cohesion Needed above MICP [kPa]",
        "figsize": (14, 11),
    },
    "required_cohesion_above_roots_micp": {
        "path": f"{PARENT_DIR}/required_cohesion_above_roots_micp_{M_PP}.tif",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "cbar_label": "Additional Cohesion Needed above Roots + MICP [kPa]",
        "figsize": (14, 11),
    },
}


def load_swiss_border():
    if not BORDERS_AVAILABLE:
        return None

    if not os.path.exists(SWISS_BORDER_PATH):
        print(f"[info] {SWISS_BORDER_PATH} not found; plotting without border")
        return None

    try:
        return gpd.read_file(SWISS_BORDER_PATH)
    except Exception as e:
        print(f"[warn] failed to load border: {e}")
        return None


def border_for_crs(swiss_border, crs):
    if swiss_border is None:
        return None

    try:
        if swiss_border.crs != crs:
            return swiss_border.to_crs(crs)

        return swiss_border

    except Exception as e:
        print(f"[warn] border reprojection failed: {e}")
        return None


def add_border(ax, swiss_border, crs):
    border = border_for_crs(swiss_border, crs)

    if border is None:
        return

    try:
        border.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )

    except Exception as e:
        print(f"[warn] border overlay failed: {e}")


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_map_with_border(name_key, config, swiss_border=None):
    if not os.path.exists(config["path"]):
        print(f"[skip] {config['path']} not found")
        return

    with rasterio.open(config["path"]) as src:
        raster = src.read(1).astype(np.float32)
        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata

    valid = np.isfinite(raster)

    if nodata is not None and np.isfinite(nodata):
        valid &= raster != nodata

    raster[~valid] = np.nan

    print(f"{name_key:36s} | valid pixels: {valid.sum():,}")

    fig, ax = plt.subplots(figsize=config["figsize"], dpi=DPI)

    im = ax.imshow(
        raster,
        cmap=config["cmap"],
        vmin=config["vmin"],
        vmax=config["vmax"],
        interpolation="nearest",
        origin="upper",
        extent=[
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ],
    )

    add_border(ax, swiss_border, crs)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
        aspect=40,
    )

    cbar.set_label(config["cbar_label"], fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    clean_axes(ax)

    ax.set_title(
        config["cbar_label"],
        fontsize=13,
        pad=15,
        fontweight="bold",
    )

    fig.tight_layout()

    out_path = f"{OUTDIR}/{name_key}.png"

    fig.savefig(
        out_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"  -> {out_path}")


def read_overlay_inputs(reference):
    fos_path = MAPS[reference]["path"]

    if not os.path.exists(fos_path):
        print(f"[skip] overlay reference not found: {fos_path}")
        return None

    if not os.path.exists(MICP_INJECTION_TIF):
        print(f"[skip] injection raster not found: {MICP_INJECTION_TIF}")
        return None

    with rasterio.open(fos_path) as src:
        fos = src.read(1).astype(np.float32)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        shape = src.shape
        nodata = src.nodata

    with rasterio.open(MICP_INJECTION_TIF) as src:
        injection = src.read(1)
        injection_shape = src.shape
        injection_transform = src.transform
        injection_crs = src.crs

    if injection_shape != shape:
        raise ValueError(f"injection shape {injection_shape} != FoS shape {shape}")

    if injection_transform != transform:
        raise ValueError("injection raster transform does not match FoS raster")

    if injection_crs != crs:
        raise ValueError(f"injection CRS {injection_crs} != FoS CRS {crs}")

    valid = np.isfinite(fos)

    if nodata is not None and np.isfinite(nodata):
        valid &= fos != nodata

    return fos, injection, valid, bounds, crs


def plot_micp_overlay(reference_key, reference, swiss_border=None):
    result = read_overlay_inputs(reference)

    if result is None:
        return

    fos, injection_raw, valid, bounds, crs = result

    injection = injection_raw > 0
    unstable = fos < 1.0

    classes = np.full(
        fos.shape,
        np.nan,
        dtype=np.float32,
    )

    classes[valid & ~injection & ~unstable] = 0

    classes[valid & ~injection & unstable] = 1

    classes[valid & injection & unstable] = 2

    classes[valid & injection & ~unstable] = 3

    n = int(np.isfinite(classes).sum())

    n0 = int((classes == 0).sum())
    n1 = int((classes == 1).sum())
    n2 = int((classes == 2).sum())
    n3 = int((classes == 3).sum())

    print("")
    print(f"MICP overlay based on {reference}:")
    print(f"0 no MICP / stable   : {n0:,} ({n0 / max(n, 1):.1%})")
    print(f"1 no MICP / unstable : {n1:,} ({n1 / max(n, 1):.1%})")
    print(f"2 MICP / unstable    : {n2:,} ({n2 / max(n, 1):.1%})")
    print(f"3 MICP / stable      : {n3:,} ({n3 / max(n, 1):.1%})")

    cmap = ListedColormap(
        [
            "white",
            "darkred",
            "forestgreen",
            "darkgreen",
        ]
    )

    cmap.set_bad(
        "white",
        alpha=0.0,
    )

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap.N,
    )

    fig, ax = plt.subplots(
        figsize=(14, 11),
        dpi=DPI,
    )

    ax.imshow(
        classes,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
        extent=[
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ],
    )

    add_border(
        ax,
        swiss_border,
        crs,
    )

    legend = [
        Patch(
            facecolor="white",
            edgecolor="black",
            label="0  No MICP / stable",
        ),
        Patch(
            facecolor="darkred",
            edgecolor="black",
            label="1  No MICP / unstable",
        ),
        Patch(
            facecolor="forestgreen",
            edgecolor="black",
            label="2  MICP possible / unstable",
        ),
        Patch(
            facecolor="darkgreen",
            edgecolor="black",
            label="3  MICP possible / stable",
        ),
    ]

    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    clean_axes(ax)

    if reference_key == "bare":
        title_reference = "Bare soil"
    else:
        title_reference = "Present-day roots"

    ax.set_title(
        f"MICP Injection Potential and Instability — {title_reference}, "
        f"$m_{{pp}}$ = {M_PP}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )

    fig.tight_layout()

    out_path = f"{OUTDIR}/" f"micp_injection_overlay_{reference_key}_{M_PP}.jpg"

    fig.savefig(
        out_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"-> {out_path}")


def main():
    swiss_border = load_swiss_border()

    for name, config in MAPS.items():
        plot_map_with_border(
            name,
            config,
            swiss_border,
        )

    for reference_key, reference in OVERLAY_REFERENCES.items():
        plot_micp_overlay(
            reference_key,
            reference,
            swiss_border,
        )

    print("")
    print(f"all maps saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
