"""
03_evaluate_bounds.py - Boundary Sensitivity & Saturation Analysis

Evaluates how changing the maximum physical limits of the drainage rate affects
the model, while keeping ET capped at a realistic 5.0 mm/day.
"""

import sys
import os
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import physics
from core import constants as const
from core import utils as ut
from validation import val_utils as autils

OUTPUT_DIR = "output/03_bounds_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
QUICK_TEST_MODE = False  # Set to False for overnight final run

CALIB_YEARS = range(2005, 2008) if QUICK_TEST_MODE else range(1991, 2026)
ALL_REGIONS = range(31, 69)

# Fix ET max at 5.0, sweep drainage max in ~0.05 granular steps
BOUNDS_TO_TEST = [
    (0.10, 5.0),
    (0.15, 5.0),
    (0.20, 5.0),
    (0.25, 5.0),
    (0.30, 5.0),
    (0.35, 5.0),
    (0.40, 5.0),
    (0.45, 5.0),
    (0.50, 5.0),  # The current baseline
]

# =====================================================================
# 1. Optimized Core Functions
# =====================================================================


def objective(params, rf_values, nfk_values, common_idx):
    d_rate, et_rate = params
    sim_array = physics.calculate_daily_saturation(
        rf_values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=nfk_values[0] * const.S_PP_ONSET_DEFAULT,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,
    )
    sim_common = sim_array[common_idx]
    pred_band = np.clip(sim_common / const.S_PP_ONSET_DEFAULT, 0.0, 1.0)
    return np.mean((pred_band - nfk_values) ** 2)


def calibrate_for_bounds(max_drain, max_et, regions_to_run):
    calib_dict = {}
    for rid in regions_to_run:
        fits = []
        try:
            avg_e, avg_n = dl.get_region_coordinates(rid)
        except:
            continue

        for yr in CALIB_YEARS:
            rf = dl.load_rainfall(avg_e, avg_n, yr)
            nfk = dl.load_bafu_moisture(rid, yr, interpolate_daily=True)
            if rf is None or len(nfk) < 5:
                continue

            common_idx = np.where(np.isin(rf.index, nfk.index))[0]
            if len(common_idx) == 0:
                continue

            res = minimize(
                objective,
                x0=[0.1, 1.5],
                args=(rf.values, nfk.loc[rf.index[common_idx]].values, common_idx),
                bounds=[(0.01, max_drain), (0.0, max_et)],
            )
            fits.append(res.x)

        if fits:
            avg = np.mean(fits, axis=0)
            calib_dict[rid] = {"drainage": avg[0], "et": avg[1]}

    return calib_dict


# =====================================================================
# 2. AUC Evaluation (Dynamic)
# =====================================================================


def min_fos_dynamic(x, y, date, calib_dict):
    x, y = ut.to_lv95(x, y)
    region_id = dl.assign_region(x, y)

    if region_id not in calib_dict:
        return np.nan

    drainage = calib_dict[region_id]["drainage"]
    et = calib_dict[region_id]["et"]

    return autils.min_fos_for_params(x, y, date, drainage, et)


def evaluate_auc_for_bounds(calib_dict):
    inv = dl.load_wsl_usable_inventory()
    if QUICK_TEST_MODE:
        inv = inv.sample(min(20, len(inv)), random_state=42).reset_index(drop=True)

    pos, neg = [], []
    for _, ev in inv.iterrows():
        f = min_fos_dynamic(ev["x"], ev["y"], ev["date"], calib_dict)
        if np.isfinite(f):
            pos.append(f)
            control = list(autils.control_dates(ev["date"], 1))[0]
            g = min_fos_dynamic(ev["x"], ev["y"], control, calib_dict)
            if np.isfinite(g):
                neg.append(g)

    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return autils.auc_score(np.array(pos), np.array(neg))


# =====================================================================
# 3. Visualization Generators
# =====================================================================


def plot_5_region_fits():
    """Generates a 5-panel subplot comparing absolute saturation across bounds."""
    print("\n--- Generating 5-Region Saturation Plot ---")

    target_regions = [32, 42, 50, 60, 65]
    test_year = 2005

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(BOUNDS_TO_TEST)))

    for idx, rid in enumerate(target_regions):
        ax = axes[idx]
        avg_e, avg_n = dl.get_region_coordinates(rid)
        rf = dl.load_rainfall(avg_e, avg_n, test_year)
        nfk = dl.load_bafu_moisture(rid, test_year, interpolate_daily=True)

        if rf is None or nfk.empty:
            ax.set_title(f"Region {rid} - Missing Data")
            continue

        common_idx = np.where(np.isin(rf.index, nfk.index))[0]

        # Plot BAFU Ground Truth scaled by Onset
        bafu_ratio = nfk / 100.0 if nfk.max() > 2.0 else nfk
        scaled_bafu = bafu_ratio * const.S_PP_ONSET_DEFAULT
        ax.plot(
            scaled_bafu.index,
            scaled_bafu.values,
            "o--",
            color="black",
            ms=3,
            label="BAFU nFK (Scaled)",
        )

        # Plot structural lines
        ax.axhline(
            const.S_PP_ONSET_DEFAULT,
            color="orange",
            ls=":",
            label=f"Onset ({const.S_PP_ONSET_DEFAULT})",
        )
        ax.axhline(
            1.0, color="gray", linestyle="--", alpha=0.3, label="Full Saturation"
        )

        # Run calibration and plot simulated saturation (S_series)
        for b_idx, (max_d, max_et) in enumerate(BOUNDS_TO_TEST):
            res = minimize(
                objective,
                x0=[0.1, 1.5],
                args=(rf.values, nfk.loc[rf.index[common_idx]].values, common_idx),
                bounds=[(0.01, max_d), (0.0, max_et)],
            )
            opt_d, opt_et = res.x

            sim_array = physics.calculate_daily_saturation(
                rf.values,
                n=const.N,
                n_perp=const.H_PERP,
                m0=nfk.iloc[0] * const.S_PP_ONSET_DEFAULT,
                s_pp_onset=const.S_PP_ONSET_DEFAULT,
                drainage_rate=opt_d,
                et_rate=opt_et,
            )

            label_str = f"Max D:{max_d} (Fitted: {opt_d:.2f})"
            ax.plot(
                rf.index,
                sim_array,
                color=colors[b_idx],
                linewidth=1.5,
                alpha=0.8,
                label=label_str,
            )

        ax.set_ylabel("Saturation Ratio")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Region {rid} (Year {test_year}) - Saturation vs Drainage Limits")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/region_saturation_subplot.png", dpi=150)
    plt.close()
    print(f"Plot saved to {OUTPUT_DIR}/region_saturation_subplot.png")


def main():
    print(f"Starting Bounds Evaluation... (Quick Test Mode: {QUICK_TEST_MODE})")

    with contextlib.redirect_stdout(io.StringIO()):
        plot_5_region_fits()

    print("\n--- Running AUC Sensitivity Sweep ---")
    auc_results = []

    for max_d, max_et in BOUNDS_TO_TEST:
        print(
            f"Calibrating system for upper limits -> Drainage: {max_d}, ET: {max_et}..."
        )
        with contextlib.redirect_stdout(io.StringIO()):
            calib_dict = calibrate_for_bounds(max_d, max_et, ALL_REGIONS)

        print("Calculating AUC...")
        auc_score = evaluate_auc_for_bounds(calib_dict)
        auc_results.append(auc_score)
        print(f"Resulting AUC: {auc_score:.3f}\n")

    labels = [f"D:{d}" for d, et in BOUNDS_TO_TEST]

    plt.figure(figsize=(8, 5))
    plt.plot(
        labels,
        auc_results,
        marker="o",
        color="darkred",
        linestyle="-",
        linewidth=2,
        markersize=8,
    )
    plt.axhline(0.70, color="gray", linestyle="--", label="AUC Target (0.70)")

    plt.title("Impact of Maximum Drainage Limits on Model Predictive Power (AUC)")
    plt.xlabel("Upper Limit of Drainage Rate (mm/day)")
    plt.ylabel("Area Under Curve (AUC)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/bounds_vs_auc.png", dpi=150)
    print(f"AUC Plot saved to {OUTPUT_DIR}/bounds_vs_auc.png")


if __name__ == "__main__":
    main()
