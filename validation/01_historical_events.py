"""
01_historical_events.py - Temporal Validation against WSL Database

This script simulates historical landslide events by extracting rainfall and soil
moisture data for the weeks surrounding a documented disaster. It validates whether
the geotechnical model correctly predicts a Factor of Safety (FoS) <= 1.0 during
the exact real-world failure window.
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core import data_loader as dl
from core import physics
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_visuals as vis

PLOT_DIR = "output/hist_plots"
RESULTS_CSV = "output/validation_results.csv"
FOS_THRESHOLD = 1.0

os.makedirs(PLOT_DIR, exist_ok=True)


def simulate_event(x, y, date):
    """Runs the physics model for a specific coordinate and time window."""
    x, y = ut.to_lv95(x, y)
    region_id, d_rate, et_rate = dl.get_region_params(x, y, auct.CALIB)

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)

    # rainfall may span a year boundary; load each needed year
    years = sorted({start.year, end.year})
    rain = dl.load_rainfall(x, y, years)

    if rain is None or rain.empty:
        return None

    rain = rain.loc[start:end]
    if rain.empty:
        return None

    bafu = dl.load_bafu_moisture(region_id, interpolate_daily=False)
    bafu_win = bafu.loc[start:end] if bafu is not None else pd.Series(dtype=float)

    # Simulate daily soil saturation
    S = physics.calculate_daily_saturation(
        rain.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=const.M0,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,  # or per-region from calibration
    )

    S_series = pd.Series(S, index=rain.index)
    m_pp = physics.pore_pressure_ratio(S_series.values, const.S_PP_ONSET_DEFAULT)

    # Compute FoS for the entire simulation period
    fos = pd.Series(
        physics.compute_fos(
            m_array=m_pp,
            c=const.C,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=const.H_V,
            beta_rad=const.beta,
            phi_rad=const.phi,
        ),
        index=S_series.index,
    )

    return rain, S_series, fos, bafu_win, region_id, d_rate, et_rate


def evaluate(fos, date):
    """Finds the minimum FoS within the defined window around the event date"""
    win = fos.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]

    if win.empty:
        return None, None

    return float(win.min()), win.idxmin().date()


def main():
    """Main Execution"""
    inv = dl.load_wsl_usable_inventory()
    print(f"Loaded {len(inv)} historical events from the WSL inventory.")

    results_data = []

    for idx, ev in inv.iterrows():
        simulation_out = simulate_event(ev["x"], ev["y"], ev["date"])

        if simulation_out is None:
            results_data.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue

        rain, S, fos, bafu_win, region_id, d_rate, et_rate = simulation_out
        min_fos, min_date = evaluate(fos, ev["date"])

        if min_fos is None:
            results_data.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue

        label = "unstable" if min_fos <= FOS_THRESHOLD else "stable"

        vis.plot_event(
            rain,
            S,
            fos,
            bafu_win,
            ev["date"],
            ev.get("municipality", "event"),
            idx,
            PLOT_DIR,
            const.S_PP_ONSET_DEFAULT,
            const.C,
        )

        win_rain = rain.loc[
            ev["date"]
            - pd.Timedelta(days=auct.WINDOW_DAYS) : ev["date"]
            + pd.Timedelta(days=auct.WINDOW_DAYS)
        ]

        results_data.append(
            {
                "municipality": ev.get("municipality", ""),
                "date": ev["date"].date(),
                "x": ev["x"],
                "y": ev["y"],
                "min_fos": round(min_fos, 3),
                "min_fos_date": min_date,
                "rain_max_window": round(
                    win_rain.max(),
                    1,
                ),
                "label": label,
                "region_id": region_id,
                "et_rate": et_rate,
                "drainage_rate": d_rate,
            }
        )

    res_df = pd.DataFrame(results_data)
    res_df.to_csv(RESULTS_CSV, index=False)

    # Print Summary Statistics
    scored = res_df[res_df["label"].isin(["stable", "unstable"])]
    n = len(scored)
    detected = (scored["label"] == "unstable").sum()

    stats_file = f"{PLOT_DIR}/historical_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as txt_stats:
        txt_stats("--- Historical Event Validation Summary ---")
        txt_stats(f"Total scored events: {n}")
        txt_stats(
            f"Overall Detection Rate: {detected}/{n} ({detected/n:.1%}) of events had FoS <= 1 within ±{auct.WINDOW_DAYS} days.\n"
        )

        txt_stats(
            "Stable events with <5mm rain in window:",
            (scored[scored.label == "stable"]["rain_max_window"] < 5).sum(),
        )

        txt_stats("\n--- Detection Rates by Rainfall Intensity Band ---")
        for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 999)]:
            band = scored[
                (scored.rain_max_window >= lo) & (scored.rain_max_window < hi)
            ]
            if len(band) > 0:
                det = (band.label == "unstable").mean()
                txt_stats(f"{lo:3d}-{hi:3d} mm: {det:4.0%} detected  (n={len(band)})")

    # Print confirmation to console
    print(f"\nModel Detection Rate: {detected}/{n} ({detected/n:.1%})")
    print(f"Validation Results CSV saved -> {RESULTS_CSV}")
    print(f"Detailed statistics logged to -> {stats_file}")
    print(f"Plots saved -> {PLOT_DIR}/")

    # Generate summary distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scored["min_fos"], bins=20, color="steelblue", edgecolor="k")
    ax.axvline(1.0, color="red", ls="--", label="Failure Threshold (FoS=1)")
    ax.set_xlabel("Minimum FoS within Event Window")
    ax.set_ylabel("Number of Historical Events")
    ax.set_title("Distribution of Minimum FoS During Landslides")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/_summary_minfos_hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

# Without calibration
#   0- 20 mm: 0% detected  (n=175)
#  20- 40 mm: 11% detected  (n=186)
#  40- 80 mm: 65% detected  (n=162)
#  80-999 mm: 98% detected  (n=115)


# With calibration
# stable events, rain in window:
# count    392.000000
# mean      23.577041
# std       15.255474
# min        0.000000
# 25%       12.800000
# 50%       22.150000
# 75%       32.700001
# max       72.500000
# Name: rain_max_window, dtype: float64

# unstable events, rain in window:
# count    246.000000
# mean      83.780488
# std       41.770840
# min       31.200001
# 25%       54.450001
# 50%       78.350002
# 75%      102.450003
# max      283.600006
# Name: rain_max_window, dtype: float64

# stable events with <5mm rain in window: 45

# Detection rate: 246/638 = 38.6% of events had FoS ≤ 1 within ±2 days.
# Results -> output/validation_results.csv | plots -> output/hist_plots/
#   0- 20 mm: 0% detected  (n=175)
#  20- 40 mm: 13% detected  (n=186)
#  40- 80 mm: 66% detected  (n=162)
#  80-999 mm: 100% detected  (n=115)
