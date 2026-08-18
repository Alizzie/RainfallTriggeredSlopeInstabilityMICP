"""
03_sweep_uphill_radius.py - Uphill Initiation Scar Search

Sweeps the uphill search radius to identify how far the initiation scar typically
sits from the recorded debris deposit in the WSL inventory.
"""

import sys
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import data_loader as dl
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

RADII = auct.RADII_M  # single source of truth (shared with the consolidated script)
OUTPUT_DIR = "output/spatial/03_uphill_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIXED_LABEL = f"Fixed {const.BETA_DEG}°"


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


def score_variant(pos, neg, min_n=1):
    """AUC + threshold metrics (Youden's J, MCC, TPR, FPR at FoS = 1) for one
    variant's pos/neg FoS arrays. Returns a dict of NaNs if there isn't enough
    data (fewer than min_n on either side).
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) < min_n or len(neg) < min_n:
        return dict(
            auc=np.nan, j=np.nan, mcc=np.nan, tpr=np.nan, fpr=np.nan, n=len(pos)
        )
    auc = autils.auc_score(pos, neg)
    event_flags, control_flags = pos <= 1.0, neg <= 1.0
    tpr, fpr = vm.tpr_fpr(event_flags, control_flags)
    j = vm.youden_j(event_flags, control_flags)
    m = vm.mcc(event_flags, control_flags)
    return dict(auc=auc, j=j, mcc=m, tpr=tpr, fpr=fpr, n=len(pos))


def main():
    inv = dl.load_wsl_usable_inventory()

    with rasterio.open(auct.DEM_TIF) as dem_src, rasterio.open(
        auct.SLOPE_TIF
    ) as slp_src:
        assert_grids_compatible(dem_src, slp_src)
        pixel_m = slp_src.res[0]
        r_max_pix = int(round(max(RADII) / pixel_m))

        # --- Pass 1: per event, angles at all radii + pore-pressure basis (bucket once) ---
        records = []  # each: (angles_dict, mpp_evt, [mpp_controls])
        for i, ev in inv.iterrows():
            x, y = dl.to_lv95(
                ev["x"], ev["y"]
            )  # LV03 -> LV95 guard (no-op if already LV95)

            angles = uphill_angles(dem_src, slp_src, x, y, RADII, r_max_pix, pixel_m)

            mpp_evt = autils.m_pp_max_at(x, y, ev["date"], auct.CALIB)
            controls = vm.find_control_date(
                ev["x"], ev["y"], ev["date"], inv, n=auct.CONTROLS_PER_EVENT
            )
            mpp_ctrls = [autils.m_pp_max_at(x, y, cd, auct.CALIB) for cd in controls]

            records.append((angles, mpp_evt, mpp_ctrls))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(inv)} processed")

    # --- Pass 2: score each radius; real vs fixed-33 on the SAME per-radius subset ---
    print(
        f"\n{'radius_m':>8} {'n':>6} {'AUC_real':>9} {'AUC_fix':>8} "
        f"{'medEvt':>7} {'medCtrl':>8}"
    )

    rows = []
    for r in RADII:
        pos_full, neg_full, pos_fix, neg_fix = [], [], [], []
        pos_hydro, neg_hydro = [], []

        for angles, mpp_evt, mpp_ctrls in records:
            angle = angles[r]
            regime = autils.classify_slope(angle)
            if regime == "invalid" or not np.isfinite(mpp_evt):
                continue
            if not mpp_ctrls or not all(np.isfinite(m) for m in mpp_ctrls):
                continue

            fos_evt = autils.fos_from_mpp(mpp_evt, angle)
            fos_fix_evt = autils.fos_from_mpp(mpp_evt, const.BETA_DEG)
            pos_full.append(fos_evt)
            pos_fix.append(fos_fix_evt)

            if regime == "hydro":
                pos_hydro.append(fos_evt)

            for m in mpp_ctrls:
                fos_ctrl = autils.fos_from_mpp(m, angle)
                neg_full.append(fos_ctrl)
                neg_fix.append(autils.fos_from_mpp(m, const.BETA_DEG))
                if regime == "hydro":
                    neg_hydro.append(fos_ctrl)

        if len(pos_full) < 20:
            print(f"{r:8d} {len(pos_full):7d}   (too few valid points, skipped)")
            continue

        full = score_variant(pos_full, neg_full)
        fix = score_variant(pos_fix, neg_fix)
        hydro = score_variant(pos_hydro, neg_hydro, min_n=10)

        rows.append(
            {
                "radius_m": r,
                "n_full": full["n"],
                "n_hydro": hydro["n"],
                "auc_full": full["auc"],
                "auc_hydro": hydro["auc"],
                "auc_fixed": fix["auc"],
                "j_full": full["j"],
                "j_hydro": hydro["j"],
                "j_fixed": fix["j"],
                "mcc_full": full["mcc"],
                "mcc_hydro": hydro["mcc"],
                "mcc_fixed": fix["mcc"],
                "tpr_full": full["tpr"],
                "fpr_full": full["fpr"],
            }
        )

    if not rows:
        print("No radius produced enough valid points to score.")
        return

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(f"{OUTPUT_DIR}/auc_vs_radius.csv", index=False)

    # --- Plot: AUC vs radius, full / hydro-only / fixed baseline ---
    def fmt(v, w):
        return f"{v:{w}.3f}" if np.isfinite(v) else f"{'n/a':>{w}}"

    print(
        f"\n{'radius_m':>8} {'n_full':>7} {'n_hydro':>8} "
        f"{'AUC_full':>9} {'AUC_hydro':>10} {'AUC_fix':>8}"
    )
    for row in rows:
        print(
            f"{row['radius_m']:8d} {row['n_full']:7d} {row['n_hydro']:8d} "
            f"{fmt(row['auc_full'], 9)} {fmt(row['auc_hydro'], 10)} {fmt(row['auc_fixed'], 8)}"
        )

    print(
        f"\n{'radius_m':>8} {'J_full':>7} {'J_hydro':>8} {'J_fix':>7} "
        f"{'MCC_full':>9} {'MCC_hydro':>10} {'MCC_fix':>8}"
    )
    for row in rows:
        print(
            f"{row['radius_m']:8d} {fmt(row['j_full'], 7)} {fmt(row['j_hydro'], 8)} "
            f"{fmt(row['j_fixed'], 7)} {fmt(row['mcc_full'], 9)} "
            f"{fmt(row['mcc_hydro'], 10)} {fmt(row['mcc_fixed'], 8)}"
        )

    # --- Plot 1: AUC vs radius, full / hydro-only / fixed baseline ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve_df["radius_m"],
        curve_df["auc_full"],
        "o-",
        color="firebrick",
        label="real angle (15 - 45°)",
    )
    plt.plot(
        curve_df["radius_m"],
        curve_df["auc_hydro"],
        "^-",
        color="darkorange",
        label="real angle (15 - 35°)",
    )
    plt.plot(
        curve_df["radius_m"],
        curve_df["auc_fixed"],
        "s--",
        color="gray",
        label=f"{FIXED_LABEL}",
    )
    bi = int(np.nanargmax(curve_df["auc_full"]))
    plt.scatter(
        [curve_df["radius_m"][bi]],
        [curve_df["auc_full"][bi]],
        s=120,
        facecolors="none",
        edgecolors="firebrick",
        linewidths=2,
        zorder=5,
        label=f"best AUC: {int(curve_df['radius_m'][bi])} m ({curve_df['auc_full'][bi]:.3f})",
    )
    if curve_df["auc_hydro"].notna().any():
        bi_h = int(curve_df["auc_hydro"].idxmax())
        plt.scatter(
            [curve_df["radius_m"][bi_h]],
            [curve_df["auc_hydro"][bi_h]],
            s=120,
            facecolors="none",
            edgecolors="darkorange",
            linewidths=2,
            zorder=5,
            label=f"best AUC: {int(curve_df['radius_m'][bi_h])} m "
            f"({curve_df['auc_hydro'][bi_h]:.3f})",
        )
    for _, row in curve_df.iterrows():
        plt.annotate(
            f"n={int(row['n_full'])}",
            (row["radius_m"], row["auc_full"]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8,
            ha="center",
            color="firebrick",
        )
    plt.title("AUC Recovery via Uphill Search Radius")
    plt.xlabel("Search Radius (m)")
    plt.ylabel("AUC Score")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/auc_vs_radius.png", dpi=150)
    plt.close()

    # --- Plot 2: threshold metrics (Youden's J solid, MCC dashed) vs radius ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for col_j, col_mcc, color, label in [
        ("j_full", "mcc_full", "firebrick", "(15 - 45°)"),
        ("j_hydro", "mcc_hydro", "darkorange", "(15 - 35°)"),
        ("j_fixed", "mcc_fixed", "gray", FIXED_LABEL),
    ]:
        ax.plot(
            curve_df["radius_m"],
            curve_df[col_j],
            "o-",
            color=color,
            label=f"J, {label}",
        )
        ax.plot(
            curve_df["radius_m"],
            curve_df[col_mcc],
            "o--",
            color=color,
            alpha=0.5,
            label=f"MCC, {label}",
        )
    if curve_df["j_full"].notna().any():
        bj = int(curve_df["j_full"].idxmax())
        ax.scatter(
            [curve_df["radius_m"][bj]],
            [curve_df["j_full"][bj]],
            s=120,
            facecolors="none",
            edgecolors="firebrick",
            linewidths=2,
            zorder=5,
            label=f"best J: {int(curve_df['radius_m'][bj])} m ({curve_df['j_full'][bj]:.3f})",
        )
    if curve_df["j_hydro"].notna().any():
        bj_h = int(curve_df["j_hydro"].idxmax())
        ax.scatter(
            [curve_df["radius_m"][bj_h]],
            [curve_df["j_hydro"][bj_h]],
            s=120,
            facecolors="none",
            edgecolors="darkorange",
            linewidths=2,
            zorder=5,
            label=f"best J: {int(curve_df['radius_m'][bj_h])} m "
            f"({curve_df['j_hydro'][bj_h]:.3f})",
        )
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_title("Threshold metrics (FoS = 1) vs uphill search radius")
    ax.set_xlabel("Search Radius (m)")
    ax.set_ylabel("score")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/threshold_metrics_vs_radius.png", dpi=150)
    plt.close(fig)

    print(f"\n-> {OUTPUT_DIR}/auc_vs_radius.png")
    print(f"-> {OUTPUT_DIR}/threshold_metrics_vs_radius.png")
    print(
        f"-> {OUTPUT_DIR}/auc_vs_radius.csv (05_compare_spatial_variants.py reads this)"
    )


if __name__ == "__main__":
    main()
