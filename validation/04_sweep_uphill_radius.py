"""
04_sweep_uphill_radius.py - Uphill Initiation Scar Search

Sweeps the uphill search radius to identify how far the initiation scar typically
sits from the recorded debris deposit in the WSL inventory.
"""

import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

RADII = auct.RADII_M  # single source of truth (shared with the consolidated script)
OUTPUT_DIR = f"{auct.OUTDIR}/04_uphill_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_raster(data, nodata):
    """Convert raster NoData sentinels to np.nan for safe math."""
    data = data.astype(float)
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def assert_grids_compatible(dem_src, slope_src, tol=1e-3):
    """Fail loudly if DEM and slope are not pixel-aligned (single-window read assumes it)."""
    if dem_src.crs != slope_src.crs:
        raise ValueError(f"CRS mismatch: DEM {dem_src.crs} vs slope {slope_src.crs}.")
    if not np.allclose(dem_src.res, slope_src.res, atol=tol):
        raise ValueError(
            f"Resolution mismatch: DEM {dem_src.res} vs slope {slope_src.res}. "
            "Point DEM_TIF at the SAME 25 m DEM the slope was derived from."
        )
    if not np.allclose(
        np.array(dem_src.transform)[:6], np.array(slope_src.transform)[:6], atol=tol
    ):
        raise ValueError("Grid origin mismatch: DEM and slope transforms differ.")


def uphill_angles(dem_src, slope_src, x, y, radii, r_max_pix, pixel_m):
    """Return {radius_m -> uphill steepest slope (deg)} for one coordinate.

    Reads a single window at the largest radius and derives every radius from it.
    r == 0 returns the raw centre pixel (deposit); r > 0 returns the steepest
    uphill, in-band pixel within the circular radius. NaN where none qualify.
    """
    out = {r: np.nan for r in radii}

    row, col = dem_src.index(x, y)
    win = Window(col - r_max_pix, row - r_max_pix, 2 * r_max_pix + 1, 2 * r_max_pix + 1)
    dem = clean_raster(
        dem_src.read(1, window=win, boundless=True, fill_value=np.nan), dem_src.nodata
    )
    slp = clean_raster(
        slope_src.read(1, window=win, boundless=True, fill_value=np.nan),
        slope_src.nodata,
    )

    center_elev = dem[r_max_pix, r_max_pix]
    center_slope = slp[r_max_pix, r_max_pix]

    # r = 0: raw single pixel (deposit), not band-limited
    if 0 in out:
        out[0] = float(center_slope) if np.isfinite(center_slope) else np.nan

    if not np.isfinite(center_elev):
        return out

    n = 2 * r_max_pix + 1
    yy, xx = np.ogrid[:n, :n]
    dist = np.sqrt((xx - r_max_pix) ** 2 + (yy - r_max_pix) ** 2) * pixel_m

    base = (
        (dem > center_elev)
        & (slp >= auct.BETA_MIN)
        & (slp <= auct.BETA_MAX)
        & np.isfinite(slp)
    )
    for r in radii:
        if r == 0:
            continue
        cand = slp[base & (dist <= r)]
        out[r] = float(cand.max()) if cand.size > 0 else np.nan
    return out


def main():
    inv = dl.load_wsl_usable_inventory()
    # inv = inv.sample(150, random_state=42).reset_index(drop=True)

    with rasterio.open(auct.DEM_TIF) as dem_src, rasterio.open(
        auct.SLOPE_TIF
    ) as slp_src:
        assert_grids_compatible(dem_src, slp_src)
        pixel_m = slp_src.res[0]
        r_max_pix = int(round(max(RADII) / pixel_m))

        # --- Pass 1: per event, angles at all radii + pore-pressure basis (bucket once) ---
        records = []  # each: (angles_dict, mpp_evt, [mpp_controls])
        for i, ev in inv.iterrows():
            x, y = ut.to_lv95(
                ev["x"], ev["y"]
            )  # LV03 -> LV95 guard (no-op if already LV95)

            angles = uphill_angles(dem_src, slp_src, x, y, RADII, r_max_pix, pixel_m)

            mpp_evt = autils.m_pp_max_at(x, y, ev["date"], auct.CALIB)
            controls = list(autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT))
            mpp_ctrls = [autils.m_pp_max_at(x, y, cd, auct.CALIB) for cd in controls]

            records.append((angles, mpp_evt, mpp_ctrls))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(inv)} processed")

    # --- Pass 2: score each radius; real vs fixed-33 on the SAME per-radius subset ---
    print(
        f"\n{'radius_m':>8} {'n':>6} {'AUC_real':>9} {'AUC_fix':>8} "
        f"{'medEvt':>7} {'medCtrl':>8}"
    )
    curve = []
    for r in RADII:
        pos_r, neg_r, pos_f, neg_f = [], [], [], []
        for angles, mpp_evt, mpp_ctrls in records:
            angle = angles[r]
            if not np.isfinite(angle) or not np.isfinite(mpp_evt):
                continue
            if not all(np.isfinite(m) for m in mpp_ctrls):
                continue

            pos_r.append(autils.fos_from_mpp(mpp_evt, angle))
            pos_f.append(autils.fos_from_mpp(mpp_evt, const.BETA_DEG))
            for m in mpp_ctrls:
                neg_r.append(autils.fos_from_mpp(m, angle))
                neg_f.append(autils.fos_from_mpp(m, const.BETA_DEG))

        if len(pos_r) < 20:
            print(f"{r:8d} {len(pos_r):6d}   (too few valid points, skipped)")
            continue

        pos_r, neg_r = np.array(pos_r), np.array(neg_r)
        pos_f, neg_f = np.array(pos_f), np.array(neg_f)
        auc_real = autils.auc_score(pos_r, neg_r)
        auc_fix = autils.auc_score(pos_f, neg_f)
        curve.append((r, len(pos_r), auc_real, auc_fix))
        print(
            f"{r:8d} {len(pos_r):6d} {auc_real:9.3f} {auc_fix:8.3f} "
            f"{np.median(pos_r):7.2f} {np.median(neg_r):8.2f}"
        )

    if not curve:
        print("No radius produced enough valid points to score.")
        return

    curve = np.array(curve, dtype=float)

    # --- Plot: AUC vs radius, real vs moving fixed baseline ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve[:, 0], curve[:, 2], "o-", color="firebrick", label="real angle (uphill)"
    )
    plt.plot(
        curve[:, 0],
        curve[:, 3],
        "s--",
        color="gray",
        label="fixed 33 deg (same subset)",
    )
    bi = int(np.argmax(curve[:, 2]))
    plt.scatter(
        [curve[bi, 0]],
        [curve[bi, 2]],
        s=120,
        facecolors="none",
        edgecolors="firebrick",
        linewidths=2,
        zorder=5,
        label=f"best: {int(curve[bi, 0])} m (AUC {curve[bi, 2]:.3f})",
    )
    for r, n, ar, af in curve:
        plt.annotate(
            f"n={int(n)}",
            (r, ar),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8,
            ha="center",
            color="firebrick",
        )
    plt.title("AUC Recovery via Uphill Search Radius")
    plt.xlabel("Search Radius (m)")
    plt.ylabel("AUC Score")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/auc_vs_radius.png", dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/auc_vs_radius.png")


if __name__ == "__main__":
    main()
