"""
01_sweep_fixed_slope.py - Fixed Baseline Slope Sweep

Sweeps an idealized uniform slope angle from 30 to 35 degrees across all historical events.
Used to find the theoretical mathematical optimum (peak AUC) without spatial map noise.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

SLOPE_ANGLES = [16, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35]
OUTPUT_DIR = "output/spatial/01_fixed_slope_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_cases():
    """Compiles the event dates and generates matching seasonal control dates."""
    inv = dl.load_wsl_usable_inventory()

    cases = []
    for _, ev in inv.iterrows():
        x, y, date = ev["x"], ev["y"], ev["date"]
        controls = vm.find_control_date(x, y, date, inv, n=auct.CONTROLS_PER_EVENT)
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

        event_flags = pos <= 1.0
        control_flags = neg <= 1.0
        tpr, fpr = vm.tpr_fpr(event_flags, control_flags)
        youden_j = vm.youden_j(event_flags, control_flags)
        mcc = vm.mcc(event_flags, control_flags)

        print(
            f"  beta={beta:>2}°  AUC={auc:.3f} (expected ~constant)  "
            f"TPR={tpr:.3f} FPR={fpr:.3f} J={youden_j:.3f} MCC={mcc:.3f}"
        )

        thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
        tpr_curve = [(pos <= t).mean() for t in thr]  # flag as landslide when FoS <= t
        fpr_curve = [(neg <= t).mean() for t in thr]

        # Stats
        results.append(
            {
                "slope_angle": beta,
                "auc": auc,
                "tpr_at_fos1": tpr,
                "fpr_at_fos1": fpr,
                "youden_j": youden_j,
                "mcc": mcc,
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
        autils.plot_roc_auc(
            fpr_curve, tpr_curve, auc, f"{OUTPUT_DIR}/roc_auc_{beta}.png"
        )
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
        label="Median Event FoS",
        marker="o",
    )
    ax.plot(
        results_df["slope_angle"],
        results_df["median_neg"],
        label="Median Control FoS",
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
        results_df["slope_angle"],
        results_df["min_pos"],
        label="Min Event FoS",
        marker="o",
    )
    ax[0].plot(
        results_df["slope_angle"],
        results_df["min_neg"],
        label="Min Control FoS",
        marker="o",
    )
    ax[0].set_xlabel("Slope Angle (degrees)")
    ax[0].set_ylabel("FoS")
    ax[0].set_title("Min FoS vs Slope Angle")
    ax[0].legend()

    ax[1].plot(
        results_df["slope_angle"],
        results_df["max_pos"],
        label="Max Event FoS",
        marker="o",
    )
    ax[1].plot(
        results_df["slope_angle"],
        results_df["max_neg"],
        label="Max Control FoS",
        marker="o",
    )
    ax[1].set_xlabel("Slope Angle (degrees)")
    ax[1].set_ylabel("FoS")
    ax[1].set_title("Max FoS vs Slope Angle")
    ax[1].legend()

    plt.savefig(f"{OUTPUT_DIR}/fos_min_max_vs_slope_angle.png")
    plt.close()

    # Youden's J and MCC vs Slope Angle
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(
        results_df["slope_angle"],
        results_df["youden_j"],
        "o-",
        color="purple",
        label="Youden's J",
    )
    ax1.plot(
        results_df["slope_angle"],
        results_df["mcc"],
        "s--",
        color="darkorange",
        label="MCC",
    )
    ax1.set_xlabel("Fixed slope angle (degrees)")
    ax1.set_ylabel("score")
    ax1.set_title("Metrics vs fixed slope angle (c'=0)\n")
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/threshold_metrics_vs_slope_angle.png", dpi=150)
    plt.close(fig)

    # AUC vs angle
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results_df["slope_angle"], results_df["auc"], "o-", color="gray")
    ax.set_ylim(
        min(0.45, results_df["auc"].min() - 0.02),
        max(0.55, results_df["auc"].max() + 0.02),
    )
    ax.set_xlabel("Fixed slope angle (degrees)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs fixed slope angle")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/auc_vs_slope_angle.png", dpi=150)
    plt.close(fig)

    best_row = results_df.loc[results_df["youden_j"].idxmax()]
    print(
        f"\nbest fixed angle by Youden's J: {int(best_row['slope_angle'])}° "
        f"(J={best_row['youden_j']:.3f}, MCC={best_row['mcc']:.3f}, "
        f"AUC={best_row['auc']:.3f} for reference)"
    )


if __name__ == "__main__":
    main()
