"""
map_saturation_sweep.py - Phase 2 item 2.1: hazard maps across the full
saturation range (m_pp = 0.0 to 1.0, 0.1 steps).

Outputs (OUTDIR):
    frames/{field}_mpp{X}.png           one map per saturation level
    saturation_sweep_summary.csv        one row per m_pp level
    saturation_sweep_log.txt            printed table, saved verbatim

Usage:
    python map_saturation_sweep.py                  # full sweep, 0.0-1.0
    python map_saturation_sweep.py --m_pp 0.5        # single level only
    python map_saturation_sweep.py --step 0.2        # coarser sweep
    python map_saturation_sweep.py --field fos_bare  # plot bare soil instead
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import physics
from core import constants as const
from core import data_loader as dl
from pipeline.utils import constants as auct

# --------------------------------------------------------------------------- config

NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"
OUTDIR = "output/impacts/03_saturation_sweep"
FRAMES_DIR = f"{OUTDIR}/frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# Same forest-class cohesion mapping used everywhere else in Phase 2 (literature
# sensitivity envelope, "mid" tier). See root_cohesion.py for the low/high sweep.
ROOT_C = {
    12: 15.0,  # closed forest
    14: 10.0,  # dense small stands (copse)
    13: 8.0,  # open forest
    6: 6.0,  # bush forest
}

BETA_MIN, BETA_MAX = auct.BETA_MIN, auct.BETA_MAX
BARE_C = const.C  # calibrated in soil_cohesion.py (~0.0 kPa)

# Plotting
PLOT_FIELDS = {
    "fos_bare": {
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "label": "Factor of Safety (bare soil)",
    },
    "fos_root": {
        "cmap": "RdYlGn",
        "vmin": 0.5,
        "vmax": 3.0,
        "label": "Factor of Safety (with roots)",
    },
    "required_cohesion": {
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "label": "Required Cohesion [kPa]",
    },
    "required_cohesion_above_roots": {
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 5.5,
        "label": "Additional Cohesion Needed Above Roots [kPa]",
    },
}
DPI = 130


# ------------------------------------------------------------------------ physics


def fos_field(beta_deg, m_pp, c_field):
    """FoS for a given slope angle, pore pressure, and cohesion field."""
    beta_rad = np.radians(beta_deg)
    return physics.compute_fos(
        c=c_field,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        beta_rad=beta_rad,
        h_v=const.H_PERP / np.cos(beta_rad),
        m_array=m_pp,
        phi_rad=const.phi,
    )


def required_cohesion(beta_deg, m_pp):
    """Cohesion [kPa] needed to exactly reach FoS = 1 at this pore-pressure ratio."""
    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)
    tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    fos0 = physics.compute_fos(
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta_rad,
        phi_rad=const.phi,
        m_array=m_pp,
    )
    return np.maximum((1.0 - fos0) * tau, 0.0)


# ---------------------------------------------------------------------- io helpers


def read_aligned(path, ref_shape, name):
    """Read a raster and check it matches a reference shape."""
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != ref_shape:
        raise ValueError(
            f"{name} shape {a.shape} does not match reference shape {ref_shape}"
        )
    return a


def cohesion_from(forest, mapping):
    """Root cohesion field from a forest classification raster and a class->kPa mapping."""
    c = np.zeros_like(forest, dtype=np.float32)
    for cls, val in mapping.items():
        c[forest == cls] = val
    return c


def to_full_grid(shape, valid, field, nodata=np.nan):
    """Expand a (valid-pixels-only) 1-D field back to the full 2-D raster shape."""
    out = np.full(shape, nodata, dtype=np.float32)
    out[valid] = field
    return out


def plot_and_save_png(array_2d, m_pp, field_name, out_path):
    """Render one field at one saturation level as a PNG (legend at bottom)."""
    cfg = PLOT_FIELDS[field_name]

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=DPI)
    fig.subplots_adjust(bottom=0.14)

    im = ax.imshow(
        array_2d,
        cmap=cfg["cmap"],
        vmin=cfg["vmin"],
        vmax=cfg["vmax"],
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.8)
    cbar.set_label(cfg["label"], fontsize=11)

    ax.set_title(
        f"{cfg['label']}  —  m_pp = {m_pp:.1f}", fontsize=13, fontweight="bold", pad=10
    )

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def load_inputs(log):
    """Load slope, forest, non-soil rasters once and build the validity mask."""
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta = src.read(1).astype(np.float32)
        slope_nd = src.nodata

    forest, _, _ = dl.load_forest_raster()
    if forest.shape != beta.shape:
        raise ValueError(f"forest shape {forest.shape} != slope shape {beta.shape}")
    nonsoil = read_aligned(NONSOIL_TIF, beta.shape, "non-soil")

    valid = np.isfinite(beta)
    if slope_nd is not None:
        valid &= beta != slope_nd
    valid &= (beta >= BETA_MIN) & (beta <= BETA_MAX)
    n_slope = int(valid.sum())
    valid &= nonsoil != 1
    n_final = int(valid.sum())

    log(f"pixels in {BETA_MIN:.0f}-{BETA_MAX:.0f} deg: {n_slope:,}")
    log(
        f"  removed as rock/glacier/water/snow: {n_slope - n_final:,} "
        f"({1 - n_final / max(n_slope, 1):.1%})"
    )
    log(f"  soil-mantled pixels used: {n_final:,}\n")

    return beta, forest, valid


def generate_at_saturation(m_pp, beta, forest, valid, plot_field, log):
    """Compute all fields at one m_pp level in memory; save one PNG; return stats."""
    b = beta[valid]
    f = forest[valid]

    c_root = cohesion_from(f, ROOT_C)
    fos_bare = fos_field(b, m_pp, BARE_C)
    fos_root = fos_field(b, m_pp, BARE_C + c_root)
    c_req = required_cohesion(b, m_pp)
    c_req_above_roots = np.maximum(c_req - (BARE_C + c_root), 0.0)

    field_map = {
        "fos_bare": fos_bare,
        "fos_root": fos_root,
        "required_cohesion": c_req,
        "required_cohesion_above_roots": c_req_above_roots,
    }

    # only the requested field is expanded to the full grid + plotted (saves memory)
    full_grid = to_full_grid(beta.shape, valid, field_map[plot_field])
    png_path = f"{FRAMES_DIR}/{plot_field}_mpp{m_pp:.1f}.png"
    plot_and_save_png(full_grid, m_pp, plot_field, png_path)
    del full_grid

    unstable_bare = float((fos_bare < 1).mean())
    unstable_root = float((fos_root < 1).mean())
    rescued = unstable_bare - unstable_root

    line = (
        f"m_pp={m_pp:.1f}  |  bare unstable {unstable_bare * 100:5.1f}%  |  "
        f"+roots unstable {unstable_root * 100:5.1f}%  |  "
        f"rescued {rescued * 100:5.1f}%"
    )
    log(line)

    return {
        "m_pp": round(m_pp, 2),
        "unstable_bare_pct": round(unstable_bare * 100, 2),
        "unstable_root_pct": round(unstable_root * 100, 2),
        "rescued_by_roots_pct": round(rescued * 100, 2),
        "median_c_req": round(float(np.median(c_req)), 3),
        "p90_c_req": round(float(np.percentile(c_req, 90)), 3),
        "max_c_req": round(float(c_req.max()), 3),
        "median_c_req_above_roots": round(float(np.median(c_req_above_roots)), 3),
        "p90_c_req_above_roots": round(float(np.percentile(c_req_above_roots, 90)), 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Saturation sweep for hazard maps (PNG only, no TIFs)."
    )
    parser.add_argument(
        "--m_pp",
        type=float,
        default=None,
        help="Single saturation level (0-1). If omitted, runs the full sweep.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Step size for the full sweep (default 0.1 -> 11 levels).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="fos_root",
        choices=list(PLOT_FIELDS),
        help="Which field to render as PNG at each level (default: fos_root).",
    )
    args = parser.parse_args()

    log_path = f"{OUTDIR}/saturation_sweep_log.txt"
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("SATURATION SWEEP")
    log("=" * 78)

    beta, forest, valid = load_inputs(log)

    if args.m_pp is not None:
        m_pp_grid = [args.m_pp]
    else:
        m_pp_grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 2)

    results = []
    for m_pp in m_pp_grid:
        results.append(
            generate_at_saturation(m_pp, beta, forest, valid, args.field, log)
        )

    if len(results) > 1:
        df = pd.DataFrame(results)
        csv_path = f"{OUTDIR}/saturation_sweep_summary.csv"
        df.to_csv(csv_path, index=False)
        log(f"\n-> {csv_path}")
        log(f"-> {FRAMES_DIR}/ ({len(results)} PNG maps, field='{args.field}')")
    else:
        log(
            f"\n-> {FRAMES_DIR}/ (single m_pp={m_pp_grid[0]:.1f} PNG, field='{args.field}')"
        )

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"-> {log_path}")


if __name__ == "__main__":
    main()
