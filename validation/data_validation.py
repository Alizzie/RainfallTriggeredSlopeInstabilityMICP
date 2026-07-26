"""
Validate the infinite-slope model against historical landslide events in Switzerland.
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

PLOT_DIR = "output/hist_plots"
RESULTS_CSV = "output/validation_results.csv"
FOS_THRESHOLD = 1.0

os.makedirs(PLOT_DIR, exist_ok=True)


def simulate_event(x, y, date):
    x, y = ut.to_lv95(x, y)
    region_id, d_rate, et_rate = ut.get_region_params(x, y, auct.CALIB)

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

    S = physics.calculate_daily_saturation(
        rain.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=const.M0,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,  # or per-region from calibration
    )
    S = pd.Series(S, index=rain.index)
    m_pp = physics.pore_pressure_ratio(S.values, const.S_PP_ONSET_DEFAULT)
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
        index=S.index,
    )
    return rain, S, fos, bafu_win, region_id, d_rate, et_rate


def evaluate(fos, date):
    win = fos.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]
    if win.empty:
        return None, None
    return float(win.min()), win.idxmin().date()


def plot_event(rain, S, fos, bafu, date, name, idx):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Top: Rainfall ---
    ax1.bar(rain.index, rain.values, color="blue", alpha=0.6)
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title(f"{name} — {date.date()}")
    ax1.grid(True, alpha=0.3)

    # --- Middle: Saturation (The Fix is Here) ---
    ax2.plot(
        S.index, S.values, color="purple", linewidth=2, label="Simulated Saturation"
    )

    if not bafu.empty:
        if bafu.max() > 2.0:
            bafu_ratio = bafu / 100.0
        else:
            bafu_ratio = bafu
        scaled_bafu = bafu_ratio.values * const.S_PP_ONSET_DEFAULT
        ax2.plot(bafu.index, scaled_bafu, "o--", color="black", ms=4, label="BAFU nFK")

    ax2.axhline(
        const.S_PP_ONSET_DEFAULT,
        color="orange",
        ls=":",
        label=f"Onset ({const.S_PP_ONSET_DEFAULT})",
    )
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.3, label="Full Saturation")
    ax2.set_ylabel("Saturation")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Bottom: Factor of Safety (Baseline Only) ---
    ax3.plot(
        fos.index,
        fos.values,
        color="red",
        linewidth=2,
        label=f"Baseline (c={const.C} kPa)",
    )
    ax3.axhline(1.0, color="gray", ls="-.", linewidth=1, label="Failure (FoS=1)")
    ax3.axvline(date, color="black", ls="--", alpha=0.5, label="Event Recorded")
    ax3.fill_between(
        fos.index, 0, fos.values, where=(fos.values <= 1.0), color="red", alpha=0.2
    )

    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Factor of Safety")
    ax3.set_ylim(0.5, 4.5)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/event_{idx:03d}_{date.date()}.png", dpi=150)
    plt.close(fig)


def main():
    inv = dl.load_wsl_inventory()
    print(f"{len(inv)} events with usable date + coordinates.")

    rows = []
    for idx, ev in inv.iterrows():
        out = simulate_event(ev["x"], ev["y"], ev["date"])
        if out is None:
            rows.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue
        rain, S, fos, bafu_win, region_id, d_rate, et_rate = out
        min_fos, min_date = evaluate(fos, ev["date"])
        if min_fos is None:
            rows.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue
        label = "unstable" if min_fos <= FOS_THRESHOLD else "stable"
        plot_event(
            rain, S, fos, bafu_win, ev["date"], ev.get("municipality", "event"), idx
        )

        win_rain = rain.loc[
            ev["date"]
            - pd.Timedelta(days=auct.WINDOW_DAYS) : ev["date"]
            + pd.Timedelta(days=auct.WINDOW_DAYS)
        ]

        rows.append(
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

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_CSV, index=False)

    scored = res[res["label"].isin(["stable", "unstable"])]
    print("stable events, rain in window:")
    print(scored[scored.label == "stable"]["rain_max_window"].describe())
    print("\nunstable events, rain in window:")
    print(scored[scored.label == "unstable"]["rain_max_window"].describe())
    print(
        "\nstable events with <5mm rain in window:",
        (scored[scored.label == "stable"]["rain_max_window"] < 5).sum(),
    )
    n = len(scored)
    detected = (scored["label"] == "unstable").sum()
    print(
        f"\nDetection rate: {detected}/{n} = {detected/n:.1%} "
        f"of events had FoS ≤ 1 within ±{auct.WINDOW_DAYS} days."
    )
    print(f"Results -> {RESULTS_CSV} | plots -> {PLOT_DIR}/")

    for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 999)]:
        band = scored[(scored.rain_max_window >= lo) & (scored.rain_max_window < hi)]
        if len(band):
            det = (band.label == "unstable").mean()
            print(f"{lo:3d}-{hi:3d} mm: {det:.0%} detected  (n={len(band)})")

    # summary: min-FoS distribution across events (the event-side of a future AUC)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scored["min_fos"], bins=20, color="steelblue", edgecolor="k")
    ax.axvline(1.0, color="red", ls="--", label="Failure threshold")
    ax.set_xlabel("Minimum FoS in event window")
    ax.set_ylabel("Events")
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
