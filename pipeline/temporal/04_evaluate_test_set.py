"""
04_evaluate_test_set.py

Evaluates the locked temporal model parameters (S_onset = 0.65, S_crit = 0.78)
against the held-out 40% test split of the WSL inventory.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const, data_loader as dl, physics
from core import val_metrics as vm

LOCKED_PARAMS_PATH = Path("output/temporal/01_calibrate_saturation/locked_params.json")
with open(LOCKED_PARAMS_PATH, "r", encoding="utf-8") as f:
    locked_params = json.load(f)
DRAINAGE_GATE = locked_params["s_onset"]
S_CRIT = locked_params["s_crit"]

EVENT_WINDOW_DAYS = 2
RAIN_START_YEAR = 1991
CONTROL_SEARCH_MAX_YEARS = 5  # how far to shift the control date to dodge a real event
OUTDIR = "output/temporal/04_evaluate_test_set"
TEST_EVENTS = "output/temporal/01_sweep_onset/test_events.csv"
os.makedirs(OUTDIR, exist_ok=True)


def load_test_events():
    path = Path(TEST_EVENTS)
    if not path.exists():
        raise FileNotFoundError(
            "Run 01_sweep_onset.py first to create the train/test split."
        )
    return pd.read_csv(path, parse_dates=["date"])


def find_control_date(row, full_df):
    """
    A same-location date ~1 year from the event with no recorded landslide
    nearby, so the control day isn't secretly a real event (label leakage)
    """
    control_date = row["date"] - pd.DateOffset(years=1)
    step = -1
    if control_date.year < RAIN_START_YEAR:
        control_date = row["date"] + pd.DateOffset(years=1)
        step = 1

    same_location = (full_df["x"] == row["x"]) & (full_df["y"] == row["y"])
    for _ in range(CONTROL_SEARCH_MAX_YEARS):
        nearby = full_df[
            same_location
            & (full_df["date"] >= control_date - pd.Timedelta(days=5))
            & (full_df["date"] <= control_date + pd.Timedelta(days=5))
        ]
        if nearby.empty:
            return control_date
        control_date -= pd.DateOffset(years=step)

    return None  # couldn't find a clean control date within the search budget


def peak_saturation(row, full_df):
    """Peak simulated saturation near the event date and its control date."""
    control_date = find_control_date(row, full_df)
    if control_date is None:
        return np.nan, np.nan, "no clean control date found"

    years_needed = {row["date"].year, control_date.year}
    start_year = max(min(years_needed) - 1, RAIN_START_YEAR)
    end_year = max(years_needed) + 1

    rain = dl.load_rainfall(row["x"], row["y"], range(start_year, end_year))
    region = dl.assign_region(row["x"], row["y"])
    if region is None:
        print(f"LOG: skipping event at ({row['x']}, {row['y']}) - outside any region")
        return np.nan, np.nan, "coordinates outside any region"
    if rain is None or rain.empty:
        return np.nan, np.nan, f"no rainfall for {start_year}-{end_year - 1}"

    drainage, et, et_amp = dl.load_calibration_params(region)
    if drainage is None:
        return np.nan, np.nan, f"no calibration params for region {region}"

    saturation = pd.Series(
        physics.calculate_daily_saturation(
            rain.to_numpy(dtype=float),
            n=const.N,
            n_perp=const.H_PERP,
            m0=DRAINAGE_GATE,
            s_pp_onset=DRAINAGE_GATE,
            drainage_rate=drainage,
            et_rate=et,
            day_of_year=rain.index.dayofyear.to_numpy(),
            et_amplitude=et_amp,
        ),
        index=rain.index,
    )

    def peak_near(date):
        window = saturation.loc[
            date
            - pd.Timedelta(days=EVENT_WINDOW_DAYS) : date
            + pd.Timedelta(days=EVENT_WINDOW_DAYS)
        ]
        return float(window.max()) if not window.empty else np.nan

    event_peak, control_peak = peak_near(row["date"]), peak_near(control_date)
    reason = (
        None
        if (np.isfinite(event_peak) and np.isfinite(control_peak))
        else "empty saturation window"
    )
    return event_peak, control_peak, reason


def evaluate_test_set(test_df, full_df):
    print(f"Evaluating {len(test_df)} hidden test events...")

    results = test_df.apply(
        lambda row: peak_saturation(row, full_df), axis=1, result_type="expand"
    )
    results.columns = ["event_saturation", "control_saturation", "drop_reason"]

    dropped = results[
        results["event_saturation"].isna() | results["control_saturation"].isna()
    ]
    if not dropped.empty:
        print(f"\nDropped {len(dropped)}/{len(test_df)} events:")
        print(dropped["drop_reason"].value_counts().to_string())

    peaks = results.dropna(subset=["event_saturation", "control_saturation"])
    print(f"\nSuccessfully simulated {len(peaks)} test events.\n")

    event_fail = peaks["event_saturation"] >= S_CRIT
    control_fail = peaks["control_saturation"] >= S_CRIT

    auc_score = vm.pairwise_auc(peaks["control_saturation"], peaks["event_saturation"])
    youden_j = vm.youden_j(event_fail, control_fail)
    mcc = vm.mcc(event_fail, control_fail)

    # Write in txt file
    with open(
        os.path.join(OUTDIR, "final_test_set_metrics.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("=" * 42 + "\n")
        f.write("      FINAL TEST SET PERFORMANCE        \n")
        f.write("=" * 42 + "\n")
        f.write(f"Locked S_onset : {DRAINAGE_GATE}\n")
        f.write(f"Locked S_crit  : {S_CRIT}\n")
        f.write("-" * 42 + "\n")
        f.write(f"AUC Score      : {auc_score:.3f}\n")
        f.write(f"Youden's J     : {youden_j:.3f}\n")
        f.write(f"MCC            : {mcc:.3f}\n")
        f.write("-" * 42 + "\n")
        f.write(f"Landslides caught (event days flagged)  : {event_fail.mean():.1%}\n")
        f.write(
            f"False alarms (control days flagged)     : {control_fail.mean():.1%}\n"
        )
        f.write("=" * 42 + "\n")

    peaks.drop(columns="drop_reason").to_csv(
        os.path.join(OUTDIR, "final_test_set_predictions.csv"), index=False
    )
    print(f"\nPredictions saved to {OUTDIR}")


def main():
    test_df = load_test_events()
    full_df = dl.load_wsl_usable_inventory()
    evaluate_test_set(test_df, full_df)


if __name__ == "__main__":
    main()
