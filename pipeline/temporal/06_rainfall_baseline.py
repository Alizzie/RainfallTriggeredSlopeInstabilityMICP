"""06_rainfall_baseline.py

Does the bucket model earn its place?

Compares the calibrated bucket model against plain antecedent rainfall sums
(no model, no parameters) at discriminating landslide days from matched
control days. Uses the same train/test split, the same control-date
screening, and the same metrics, so the only difference is the predictor.

If cumulative rainfall alone reaches a similar AUC, the bucket model and its
calibrated parameters are adding nothing over raw rainfall.
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

WINDOWS = [1, 3, 7, 14, 30, 60]  # antecedent rainfall sums to test [days]
EVENT_WINDOW_DAYS = 2
RAIN_START_YEAR = 1991
TEST_EVENTS = "output/temporal/01_sweep_onset/test_events.csv"
OUTDIR = Path("output/temporal/06_rainfall_baseline")
OUTDIR.mkdir(parents=True, exist_ok=True)

SCREENING_LOADER = dl.load_wsl_usable_inventory

LOCKED_PARAMS_PATH = Path("output/temporal/01_calibrate_saturation/locked_params.json")
with open(LOCKED_PARAMS_PATH, "r", encoding="utf-8") as f:
    locked_params = json.load(f)
S_ONSET = locked_params["s_onset"]


def load_test_events():
    path = Path(TEST_EVENTS)
    if not path.exists():
        raise FileNotFoundError("Run 01_sweep_onset.py first to create the split.")
    return pd.read_csv(path, parse_dates=["date"])


def predictors_for_event(row, control_date):
    """Bucket saturation and antecedent rainfall sums at the event and control
    date. Returns two dicts (event, control) or None if data is unavailable."""
    years = {row["date"].year, control_date.year}
    rain = dl.load_rainfall(
        row["x"], row["y"], range(max(min(years) - 1, RAIN_START_YEAR), max(years) + 1)
    )
    region = dl.assign_region(row["x"], row["y"])
    if rain is None or rain.empty or region is None:
        return None
    drainage, et, et_amp = dl.load_calibration_params(region)
    if drainage is None:
        return None

    saturation = pd.Series(
        physics.calculate_daily_saturation(
            rain.to_numpy(dtype=float),
            n=const.N,
            n_perp=const.H_PERP,
            m0=S_ONSET,
            s_pp_onset=S_ONSET,
            drainage_rate=drainage,
            et_rate=et,
            day_of_year=rain.index.dayofyear.to_numpy(),
            et_amplitude=et_amp,
        ),
        index=rain.index,
    )
    # Rolling antecedent rainfall totals, one column per window length.
    sums = {w: rain.rolling(f"{w}D").sum() for w in WINDOWS}

    def at(date):
        lo = date - pd.Timedelta(days=EVENT_WINDOW_DAYS)
        hi = date + pd.Timedelta(days=EVENT_WINDOW_DAYS)
        sat_window = saturation.loc[lo:hi]
        if sat_window.empty:
            return None
        out = {"bucket_saturation": float(sat_window.max())}
        for w, series in sums.items():
            window = series.loc[lo:hi]
            out[f"rain_{w}d"] = float(window.max()) if not window.empty else np.nan
        return out

    event, control = at(row["date"]), at(control_date)
    return None if event is None or control is None else (event, control)


def build_table(events):
    full_inventory = SCREENING_LOADER()
    event_rows, control_rows = [], []
    for _, row in events.iterrows():
        control_date = vm.find_control_date(
            row["x"],
            row["y"],
            row["date"],
            full_inventory,
            rain_start_year=RAIN_START_YEAR,
        )
        if control_date is None:
            continue
        result = predictors_for_event(row, control_date)
        if result is None:
            continue
        event_rows.append(result[0])
        control_rows.append(result[1])

    print(f"{len(event_rows)}/{len(events)} test events usable")
    return pd.DataFrame(event_rows), pd.DataFrame(control_rows)


def evaluate(event_df, control_df):
    """AUC and best-threshold Youden's J / MCC for every predictor.

    All predictors here are 'higher = more landslide-like', so scores are
    negated for pairwise_auc, which expects lower = more landslide-like.
    """
    rows = []
    for column in event_df.columns:
        e = event_df[column].to_numpy(dtype=float)
        c = control_df[column].to_numpy(dtype=float)
        mask = np.isfinite(e) & np.isfinite(c)
        e, c = e[mask], c[mask]
        if e.size == 0:
            continue

        auc = vm.pairwise_auc(-e, -c)
        # Best achievable J/MCC over all thresholds, so each predictor is
        # judged at its own optimum rather than a threshold chosen for another.
        candidates = np.quantile(np.concatenate([e, c]), np.linspace(0.01, 0.99, 99))
        best_j, best_mcc, best_t = -np.inf, np.nan, np.nan
        for t in candidates:
            j = vm.youden_j(e >= t, c >= t)
            if j > best_j:
                best_j, best_mcc, best_t = j, vm.mcc(e >= t, c >= t), t
        rows.append(
            {
                "predictor": column,
                "auc": auc,
                "best_youden_j": best_j,
                "mcc_at_best_j": best_mcc,
                "threshold": best_t,
                "n": int(e.size),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def main():
    events = load_test_events()
    event_df, control_df = build_table(events)
    summary = evaluate(event_df, control_df)
    summary.to_csv(OUTDIR / "rainfall_baseline_summary.csv", index=False)

    with open(OUTDIR / "rainfall_baseline_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 42 + "\n")
        f.write("      RAINFALL-ONLY BASELINE        \n")
        f.write("=" * 42 + "\n")
        f.write(f"Locked S_onset : {S_ONSET}\n")
        f.write("-" * 42 + "\n")
        f.write(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        f.write("\n\n")

        bucket = summary[summary["predictor"] == "bucket_saturation"]["auc"].iloc[0]
        best_rain = summary[summary["predictor"] != "bucket_saturation"].iloc[0]
        f.write(f"Bucket model AUC        : {bucket:.3f}")
        f.write(
            f"Best rainfall-only AUC  : {best_rain['auc']:.3f}  ({best_rain['predictor']})"
        )
        f.write(f"Difference              : {bucket - best_rain['auc']:+.3f}")
        f.write("\n")
        if bucket - best_rain["auc"] < 0.02:
            f.write(
                "-> The bucket model adds little value over raw antecedent rainfall."
            )
        else:
            f.write("-> The bucket model outperforms raw antecedent rainfall.")
        f.write(f"\nSaved: {OUTDIR / 'rainfall_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
