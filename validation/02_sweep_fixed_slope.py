"""
02_sweep_fixed_slope.py - Fixed Baseline Slope Sweep

Sweeps an idealized uniform slope angle from 30 to 35 degrees across all historical events.
Used to find the theoretical mathematical optimum (peak AUC) without spatial map noise.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from validation import val_constants as auct
from validation import val_utils as autils

SLOPE_ANGLES = [35, 36]
OUTPUT_DIR = f"{auct.OUTDIR}/02_fixed_slope_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_cases():
    """Compiles the event dates and generates matching seasonal control dates."""
    inv = dl.load_wsl_usable_inventory()

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


def main():
    """Executes the AUC evaluation for a specific assumed slope angle."""
    cases = prepare_cases()

    results = []

    for beta in SLOPE_ANGLES:
        print(f"Sweeping Angle: {beta}°...")

        pos, neg = [], []

        for i, case in enumerate(cases):
            f = autils.min_fos_at(case["x"], case["y"], case["date"], beta)
            if not np.isfinite(f):
                continue

            pos.append(f)

            for control_date in case["control_dates"]:
                g = autils.min_fos_at(case["x"], case["y"], control_date, beta)
                if np.isfinite(g):
                    neg.append(g)

            if (i + 1) % 50 == 0:
                print(
                    f"  {i + 1}/{len(cases)} processed  (pos={len(pos)}, neg={len(neg)})"
                )

        pos, neg = np.array(pos), np.array(neg)
        if len(pos) == 0 or len(neg) == 0:
            print("not enough data to score")
            continue

        auc = autils.auc_score(pos, neg)
        print(f"  AUC for slope {beta}°: {auc:.3f}")

        thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
        tpr = [(pos <= t).mean() for t in thr]  # flag as landslide when FoS <= t
        fpr = [(neg <= t).mean() for t in thr]

        # Stats
        results.append(
            {
                "slope_angle": beta,
                "auc": auc,
                "len_pos": len(pos),
                "len_neg": len(neg),
                "median_pos": np.median(pos),
                "median_neg": np.median(neg),
                "mean_pos": np.mean(pos),
                "mean_neg": np.mean(neg),
                "min_pos": np.abs(pos).min(),
                "min_neg": np.abs(neg).min(),
                "max_pos": np.abs(pos).max(),
                "max_neg": np.abs(neg).max(),
            }
        )

        # Generate and save statistical plots
        autils.plot_roc_auc(fpr, tpr, auc, f"{OUTPUT_DIR}/roc_auc_{beta}.png")
        autils.plot_fos_distribution(
            pos,
            neg,
            auc,
            f"{OUTPUT_DIR}/fos_distributions_{beta}.png",
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/sweep_results.csv", index=False)

    # Plot FoS statistics across slope angles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        results_df["slope_angle"],
        results_df["median_pos"],
        label="Median Pos",
        marker="o",
    )
    ax.plot(
        results_df["slope_angle"],
        results_df["median_neg"],
        label="Median Neg",
        marker="o",
    )
    ax.set_xlabel("Slope Angle (degrees)")
    ax.set_ylabel("FoS")
    ax.set_title("Median FoS vs Slope Angle")
    ax.legend()
    plt.savefig(f"{OUTPUT_DIR}/fos_vs_slope_angle.png")
    plt.close()

    # Plot FoS Min/Max statistics across slope angles (two subplots)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(
        results_df["slope_angle"], results_df["min_pos"], label="Min Pos", marker="o"
    )
    ax[0].plot(
        results_df["slope_angle"], results_df["min_neg"], label="Min Neg", marker="o"
    )
    ax[0].set_xlabel("Slope Angle (degrees)")
    ax[0].set_ylabel("FoS")
    ax[0].set_title("Min FoS vs Slope Angle")
    ax[0].legend()

    ax[1].plot(
        results_df["slope_angle"], results_df["max_pos"], label="Max Pos", marker="o"
    )
    ax[1].plot(
        results_df["slope_angle"], results_df["max_neg"], label="Max Neg", marker="o"
    )
    ax[1].set_xlabel("Slope Angle (degrees)")
    ax[1].set_ylabel("FoS")
    ax[1].set_title("Max FoS vs Slope Angle")
    ax[1].legend()

    plt.savefig(f"{OUTPUT_DIR}/fos_min_max_vs_slope_angle.png")
    plt.close()


if __name__ == "__main__":
    main()
