"""
02_auc_statistics.py - ROC & AUC Model Sensitivity Analysis

This script conducts a statistical evaluation of the physical model's predictive power.
It calculates the Area Under the Curve (AUC) by comparing the Factor of Safety (FoS)
during documented landslide events (positives) against randomly selected, non-event
control dates (negatives) at the exact same geographic coordinates and seasons.

Target AUC: > 0.70 (indicating strong predictive discrimination capability).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core import data_loader as dl
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

SLOPE_ANGLES = [i for i in range(30, 35)]
OUTPUT_DIR = f"{auct.OUTDIR}/fix_slope"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def min_fos_at(x, y, date, beta_deg=const.BETA_DEG):
    """ "
    Calculates the minimum FoS within the defined time window around a given date.
    Returns NaN if insufficient historical rainfall data exists.
    """
    x, y = ut.to_lv95(x, y)
    _, drainage, et = dl.get_region_params(x, y, auct.CALIB)

    if drainage is None:
        return np.nan

    return autils.min_fos_for_params(x, y, date, drainage, et, beta_deg=beta_deg)


def prepare_cases():
    """Compiles the event dates and generates matching seasonal control dates."""
    inv = dl.load_wsl_usable_inventory()

    if auct.MAX_EVENTS:
        inv = inv.sample(min(auct.MAX_EVENTS, len(inv)), random_state=0).reset_index(
            drop=True
        )

    cases = []
    for _, ev in inv.iterrows():
        controls = list(autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT))
        cases.append(
            {
                "x": ev["x"],
                "y": ev["y"],
                "date": ev["date"],
                "control_dates": controls,
            }
        )

    return cases


def run(beta_deg, cases):
    """Executes the AUC evaluation for a specific assumed slope angle."""
    print(
        f"Evaluating {len(cases)} events with {auct.CONTROLS_PER_EVENT} matched controls each..."
    )

    pos, neg = [], []

    for i, case in enumerate(cases):
        f = min_fos_at(case["x"], case["y"], case["date"], beta_deg)
        if not np.isfinite(f):
            continue
        pos.append(f)

        for control_date in case["control_dates"]:
            g = min_fos_at(case["x"], case["y"], control_date, beta_deg)
            if np.isfinite(g):
                neg.append(g)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)} processed  (pos={len(pos)}, neg={len(neg)})")

    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0:
        print("not enough data to score")
        return

    auc = autils.auc_score(pos, neg)

    thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
    tpr = [(pos <= t).mean() for t in thr]  # flag as landslide when FoS <= t
    fpr = [(neg <= t).mean() for t in thr]

    with open(f"{OUTPUT_DIR}/roc_data_{beta_deg}.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- Statistics for Slope Angle: {beta_deg}° ---")
        f.write(f"\nValid Positives: {len(pos)} | Controls: {len(neg)}")
        f.write(
            f"\nMedian FoS - Events: {np.median(pos):.2f} | Controls: {np.median(neg):.2f}"
        )
        f.write(
            f"\nMin FoS - Events: {np.abs(pos).min():.2f} | Controls: {np.abs(neg).min():.2f}"
        )
        f.write(
            f"\nMax FoS - Events: {np.abs(pos).max():.2f} | Controls: {np.abs(neg).max():.2f}"
        )
        f.write(f"\nCalculated AUC: {auc:.3f} (Target > 0.70)\n")

    # Generate and save statistical plots
    autils.plot_roc_auc(fpr, tpr, auc, f"{OUTPUT_DIR}/roc_auc_{beta_deg}.png")
    autils.plot_fos_distribution(
        pos,
        neg,
        auc,
        f"{OUTPUT_DIR}/fos_distributions_{beta_deg}.png",
    )


if __name__ == "__main__":
    cases = prepare_cases()

    for beta in SLOPE_ANGLES:
        print(f"\n{'='*40}\nRunning Analysis for Slope Angle: {beta}°\n{'='*40}")
        run(beta, cases)
