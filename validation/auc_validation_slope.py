"""
auc_validation_slope.py — AUC with REAL slope angles vs the fixed 33 deg.

Extends auc_validation.py: instead of const.beta for every point, it samples the
local slope from the 10 m raster at each landslide/control location, two ways:
  - single  : the exact pixel at the coordinate
  - max3x3  : the steepest pixel in a 3x3 window (robust to the ~1-pixel coordinate
              uncertainty of the inventory, which often marks the deposit, not the scar)

Three variants are compared on the SAME point set (angles in 15-45 deg for both
methods, so the infinite-slope model is valid): fixed 33, single, max3x3.

Efficiency: the bucket run does not depend on slope, so per point+date we compute the
max pore-pressure once (min FoS in the window = FoS at max pore pressure), then evaluate
FoS for each angle from that same basis.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

_SRC = rasterio.open(auct.SLOPE_TIF)
_SLOPE_CACHE = {}


def sample_angles(x, y):
    """Returns (single_pixel_deg, max_3x3_deg) at (easting, northing). NaN if off-raster."""
    key = (round(x), round(y))
    if key in _SLOPE_CACHE:
        return _SLOPE_CACHE[key]
    try:
        row, col = _SRC.index(x, y)
        a = _SRC.read(
            1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan
        ).astype("float32")
    except Exception:
        _SLOPE_CACHE[key] = (np.nan, np.nan)
        return _SLOPE_CACHE[key]
    if _SRC.nodata is not None:
        a[a == _SRC.nodata] = np.nan
    single = a[1, 1]
    mx = np.nanmax(a) if np.isfinite(a).any() else np.nan
    res = (float(single), float(mx))
    _SLOPE_CACHE[key] = res
    return res


def main():
    inv = dl.load_wsl_usable_inventory()
    if auct.MAX_EVENTS:
        inv = inv.sample(min(auct.MAX_EVENTS, len(inv)), random_state=0).reset_index(
            drop=True
        )
    print(f"{len(inv)} events; {auct.CONTROLS_PER_EVENT} controls each...")

    recs = []  # each: label, m_pp_max, single_deg, max3_deg
    for i, ev in inv.iterrows():
        x, y = ut.to_lv95(ev["x"], ev["y"])
        single_deg, max3_deg = sample_angles(x, y)
        mpp = autils.m_pp_max_at(x, y, ev["date"], auct.CALIB)
        if np.isfinite(mpp):
            recs.append((1, mpp, single_deg, max3_deg))
        for cd in autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT):
            g = autils.m_pp_max_at(x, y, cd, auct.CALIB)
            if np.isfinite(g):
                recs.append((0, g, single_deg, max3_deg))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(inv)} processed  (records={len(recs)})")

    df = pd.DataFrame(recs, columns=["label", "mpp", "single_deg", "max3_deg"])

    # common valid subset: both angle methods inside 15-45 deg
    in_range = lambda s: s.between(auct.BETA_MIN, auct.BETA_MAX)
    valid = df.dropna(subset=["single_deg", "max3_deg"])
    valid = valid[in_range(valid["single_deg"]) & in_range(valid["max3_deg"])]
    print(
        f"\nrecords: {len(df)} | valid (angles 15-45 deg both methods): {len(valid)} "
        f"| dropped: {len(df) - len(valid)}"
    )

    variants = {
        "fixed 33 deg": valid["mpp"].apply(
            lambda m: autils.fos_from_mpp(m, const.BETA_DEG)
        ),
        "single pixel": valid.apply(
            lambda r: autils.fos_from_mpp(r["mpp"], r["single_deg"]), axis=1
        ),
        "max 3x3": valid.apply(
            lambda r: autils.fos_from_mpp(r["mpp"], r["max3_deg"]), axis=1
        ),
    }
    lab = valid["label"].values

    print(f"\n{'variant':>14} {'AUC':>7} {'medFoS_evt':>11} {'medFoS_ctrl':>12}")
    results = {}
    for name, fos in variants.items():
        pos, neg = fos.values[lab == 1], fos.values[lab == 0]
        auc = autils.auc_score(pos, neg)
        results[name] = (auc, pos, neg)
        print(f"{name:>14} {auc:7.3f} {np.median(pos):11.2f} {np.median(neg):12.2f}")

    # --- plot 1: ROC overlay ---
    colors = {
        "fixed 33 deg": "gray",
        "single pixel": "steelblue",
        "max 3x3": "firebrick",
    }
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, (a, pos, neg) in results.items():
        fpr, tpr = autils.roc(pos, neg)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name}: AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", alpha=0.4, label="random (0.50)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC — fixed vs real slope angle")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{auct.OUTDIR}/roc_slope_compare.png", dpi=150)
    plt.close(fig)

    # --- plot 2: separation (max3x3) + control fan-out ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    _, pos3, neg3 = results["max 3x3"]
    bins = np.linspace(0, 4, 41)
    a1.hist(
        neg3, bins=bins, density=True, alpha=0.5, color="steelblue", label="controls"
    )
    a1.hist(pos3, bins=bins, density=True, alpha=0.5, color="firebrick", label="events")
    a1.axvline(1.0, color="black", ls="--", alpha=0.6, label="FoS = 1")
    a1.set_xlabel("min FoS in window")
    a1.set_ylabel("density")
    a1.set_title(f"real angle (max 3x3): AUC={results['max 3x3'][0]:.3f}")
    a1.legend()

    _, _, neg_fixed = results["fixed 33 deg"]
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
    fig.savefig(f"{auct.OUTDIR}/fos_slope_fanout.png", dpi=150)
    plt.close(fig)

    print(
        f"\n-> {auct.OUTDIR}/roc_slope_compare.png | {auct.OUTDIR}/fos_slope_fanout.png"
    )


if __name__ == "__main__":
    main()
