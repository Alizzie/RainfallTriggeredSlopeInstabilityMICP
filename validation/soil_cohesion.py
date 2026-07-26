"""
calibrate_cohesion.py — J5: calibrate the effective soil cohesion C.

DATA: the case-control set (same location, same calendar month, other years) that gave the
clean temporal AUC 0.843 — preferred over the real-angle spatial set, whose discrimination
is degraded by deposit-vs-scar location uncertainty (shown earlier; Leonarduzzi 2021).

Per point we need only the MAX pore-pressure ratio in the window (min FoS = FoS at max m_pp),
so J(C) is a cheap scan. Best C = the operating point on the ROC that best separates
landslide days from control days, expressed as a cohesion.

PLAUSIBILITY (literature_anchoring.md): soils here are mostly GM/SM (low cohesion);
Leonarduzzi treat soil c=0 and add root cohesion 5-22 kPa. So an effective soil c of a few
kPa is expected; a large value would be a red flag.

Outputs: youden_vs_C.png, fos_dist_bestC.png, cohesion_scan.csv, and the best C.
"""

import sys
import os
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const
from core import utils as ut
from validation import val_constants as vct
from validation import val_utils as vut


def max_mpp(x, y, date):
    """Get max pore-pressure ratio in the +- window around date at (x, y). NaN if no data."""
    _, dr, et = ut.get_region_params(x, y, vct.CALIB)
    start = date - pd.Timedelta(days=vct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=vct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None:
        return np.nan
    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        S = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=dr,
            et_rate=et,
        )
    mpp = pd.Series(
        physics.pore_pressure_ratio(S, const.S_PP_ONSET_DEFAULT), index=rain.index
    )
    w = mpp.loc[
        date
        - pd.Timedelta(days=vct.WINDOW_DAYS) : date
        + pd.Timedelta(days=vct.WINDOW_DAYS)
    ]
    return float(w.max()) if not w.empty else np.nan


def fos_fixed(mpp, c):
    """FoS at the fixed 33 deg slope for a pore-pressure array and cohesion c."""
    return physics.compute_fos(
        m_array=mpp,
        c=c,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )


def auc(pos_score, neg_score):
    """Higher score = more landslide-like (here: m_pp)."""
    s = np.concatenate([pos_score, neg_score])
    r = pd.Series(s).rank(method="average").values
    rp = r[: len(pos_score)].sum()
    return (rp - len(pos_score) * (len(pos_score) + 1) / 2) / (
        len(pos_score) * len(neg_score)
    )


def main():
    stats_txt = open(f"{vct.OUTDIR}/cohesion_stats.txt", "w", encoding="utf-8")
    sys.stdout = stats_txt
    sys.stderr = stats_txt

    inv = dl.load_wsl_usable_inventory()
    if vct.MAX_EVENTS:
        inv = inv.sample(min(vct.MAX_EVENTS, len(inv)), random_state=0).reset_index(
            drop=True
        )
    print(f"{len(inv)} events; {vct.CONTROLS_PER_EVENT} controls each...")

    pos, neg = [], []
    for i, ev in inv.iterrows():
        x, y = ut.to_lv95(ev["x"], ev["y"])
        m = max_mpp(x, y, ev["date"])
        if np.isfinite(m):
            pos.append(m)
        for cd in vut.control_dates(ev["date"], vct.CONTROLS_PER_EVENT):
            g = max_mpp(x, y, cd)
            if np.isfinite(g):
                neg.append(g)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(inv)}  (pos={len(pos)}, neg={len(neg)})")

    pos, neg = np.array(pos), np.array(neg)
    a = auc(pos, neg)  # ranking quality — should match the ~0.843 from before

    rows = []
    for c in vct.C_GRID:
        fp, fn = fos_fixed(pos, c), fos_fixed(neg, c)
        tpr, fpr = float(np.mean(fp < 1.0)), float(np.mean(fn < 1.0))
        rows.append({"C": c, "tpr": tpr, "fpr": fpr, "J": tpr - fpr})
    scan = pd.DataFrame(rows)
    best = scan.loc[scan["J"].idxmax()]
    scan.to_csv(f"{vct.OUTDIR}/cohesion_scan.csv", index=False)

    print(f"\npositives {len(pos)} | controls {len(neg)}")
    print(f"AUC (ranking, invariant to C): {a:.3f}   (sanity: should be ~0.84)")
    print(f"BEST C = {best.C:.2f} kPa   (current constants.C = {const.C})")
    print(f"  TPR {best.tpr:.1%} | FPR {best.fpr:.1%} | Youden J {best.J:.3f}")
    print(
        "  plausibility: GM/SM soils + Leonarduzzi (soil c=0 + roots) -> a few kPa expected"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan.C, scan.J, color="purple", lw=2, label="Youden J = TPR - FPR")
    ax.plot(
        scan.C,
        scan.tpr,
        color="firebrick",
        ls="--",
        alpha=0.6,
        label="TPR (events < 1)",
    )
    ax.plot(
        scan.C,
        scan.fpr,
        color="steelblue",
        ls="--",
        alpha=0.6,
        label="FPR (controls < 1)",
    )
    ax.axvline(best.C, color="black", ls=":", label=f"best C = {best.C:.2f} kPa")
    ax.axvline(const.C, color="gray", ls=":", alpha=0.5, label=f"current C = {const.C}")
    ax.set_xlabel("cohesion C (kPa)")
    ax.set_ylabel("rate")
    ax.set_title(f"Cohesion calibration (AUC ceiling {a:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{vct.OUTDIR}/youden_vs_C.png", dpi=150)
    plt.close(fig)

    fp, fn = fos_fixed(pos, best.C), fos_fixed(neg, best.C)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 4, 41)
    ax.hist(fn, bins=bins, density=True, alpha=0.5, color="steelblue", label="controls")
    ax.hist(fp, bins=bins, density=True, alpha=0.5, color="firebrick", label="events")
    ax.axvline(1.0, color="black", ls="--", label="FoS = 1")
    ax.set_xlabel("min FoS in window")
    ax.set_ylabel("density")
    ax.set_title(f"FoS at calibrated C = {best.C:.2f} kPa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{vct.OUTDIR}/fos_dist_bestC.png", dpi=150)
    plt.close(fig)

    print(
        f"\n-> set constants.C = {best.C:.2f}, then rerun spatial_fos.py; plots in {vct.OUTDIR}/"
    )

    stats_txt.close()


if __name__ == "__main__":
    main()
