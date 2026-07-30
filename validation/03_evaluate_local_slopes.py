"""
03_evaluate_local_slopes.py — AUC with REAL slope angles vs the fixed 33 deg.

Instead of const.BETA_DEG for every point, it samples the local slope from the
25 m raster at each landslide/control location, two ways:
  - single  : the exact pixel at the coordinate
  - max3x3  : the steepest pixel in a 3x3 window (robust to the ~1-pixel coordinate
              uncertainty of the inventory, which often marks the deposit, not the scar)

Three variants are compared on the SAME point set: fixed 33, single, max3x3.
Only points whose sampled angle is inside the infinite-slope validity band
[15, 45] deg contribute for the real-angle variants.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

_SLOPE_CACHE = {}
OUTPUT_DIR = f"{auct.OUTDIR}/03_local_slope_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_slopes(src, x, y):
    """Return (single_pixel_deg, max_3x3_deg) at (easting, northing). NaN if off-raster."""
    key = (round(x), round(y))
    if key in _SLOPE_CACHE:
        return _SLOPE_CACHE[key]
    try:
        row, col = src.index(x, y)
        a = src.read(
            1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan
        ).astype("float32")
    except Exception:
        _SLOPE_CACHE[key] = (np.nan, np.nan)
        return _SLOPE_CACHE[key]
    if src.nodata is not None:
        a[a == src.nodata] = np.nan
    single = float(a[1, 1])
    mx = float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
    res = (single, mx)
    _SLOPE_CACHE[key] = res
    return res


def in_band(angle_deg):
    """True if the angle is inside the infinite-slope validity band [15, 45] deg."""
    return np.isfinite(angle_deg) and auct.BETA_MIN <= angle_deg <= auct.BETA_MAX


def main():
    inv = dl.load_wsl_usable_inventory()

    res = {
        "Single Pixel": {"pos": [], "neg": []},
        "Max 3x3": {"pos": [], "neg": []},
        "Fixed 33 deg": {"pos": [], "neg": []},
    }

    with rasterio.open(auct.SLOPE_TIF) as src:
        for i, ev in inv.iterrows():
            x, y = ut.to_lv95(ev["x"], ev["y"])
            single, max3 = get_slopes(src, x, y)

            # angle per variant (Fixed always valid; real angles must be in-band)
            angles = {
                "Single Pixel": single if in_band(single) else np.nan,
                "Max 3x3": max3 if in_band(max3) else np.nan,
                "Fixed 33 deg": const.BETA_DEG,
            }

            # --- positive (event) ---
            for name, beta in angles.items():
                if not np.isfinite(beta):
                    continue
                f = autils.min_fos_at(x, y, ev["date"], beta_deg=beta)
                if np.isfinite(f):
                    res[name]["pos"].append(f)

            # --- negatives (matched controls) ---
            for control_date in autils.control_dates(
                ev["date"], auct.CONTROLS_PER_EVENT
            ):
                for name, beta in angles.items():
                    if not np.isfinite(beta):
                        continue
                    g = autils.min_fos_at(x, y, control_date, beta_deg=beta)
                    if np.isfinite(g):
                        res[name]["neg"].append(g)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(inv)} processed")

    # --- AUC table ---
    print(f"\n{'variant':>14} {'AUC':>7} {'medFoS_evt':>11} {'medFoS_ctrl':>12}")
    results = {}
    for name, data in res.items():
        pos, neg = np.array(data["pos"]), np.array(data["neg"])
        if len(pos) == 0 or len(neg) == 0:
            print(f"{name:>14}  insufficient data")
            continue
        auc = autils.auc_score(pos, neg)
        results[name] = (auc, pos, neg)
        print(f"{name:>14} {auc:7.3f} {np.median(pos):11.2f} {np.median(neg):12.2f}")

    # --- plot 1: ROC overlay ---
    colors = {
        "Fixed 33 deg": "gray",
        "Single Pixel": "steelblue",
        "Max 3x3": "firebrick",
    }
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, (auc, pos, neg) in results.items():
        fpr, tpr = autils.roc(pos, neg)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name}: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", alpha=0.4, label="random (0.50)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC — fixed vs real slope angle")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_slope_compare.png", dpi=150)
    plt.close(fig)

    # --- plot 2: separation (max 3x3) + control fan-out ---
    if "Max 3x3" in results and "Fixed 33 deg" in results:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        _, pos3, neg3 = results["Max 3x3"]
        bins = np.linspace(0, 4, 41)
        a1.hist(
            neg3,
            bins=bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="controls",
        )
        a1.hist(
            pos3, bins=bins, density=True, alpha=0.5, color="firebrick", label="events"
        )
        a1.axvline(1.0, color="black", ls="--", alpha=0.6, label="FoS = 1")
        a1.set_xlabel("min FoS in window")
        a1.set_ylabel("density")
        a1.set_title(f"real angle (max 3x3): AUC={results['Max 3x3'][0]:.3f}")
        a1.legend()

        _, _, neg_fixed = results["Fixed 33 deg"]
        a2.hist(
            neg_fixed,
            bins=bins,
            density=True,
            alpha=0.5,
            color="gray",
            label="fixed 33 deg",
        )
        a2.hist(
            neg3,
            bins=bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="real angle (max 3x3)",
        )
        a2.axvline(1.0, color="black", ls="--", alpha=0.6)
        a2.set_xlabel("min FoS in window (controls only)")
        a2.set_ylabel("density")
        a2.set_title("controls: fixed vs real angle (fan-out)")
        a2.legend()

        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/fos_slope_fanout.png", dpi=150)
        plt.close(fig)

    print(
        f"\n-> {OUTPUT_DIR}/roc_slope_compare.png | {OUTPUT_DIR}/fos_slope_fanout.png"
    )


if __name__ == "__main__":
    main()
