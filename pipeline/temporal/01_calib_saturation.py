"""calibrate_saturation.py

Finds the soil-saturation level above which a landslide becomes likely, by
comparing peak saturation on days with a recorded landslide against the same
location one year earlier (no landslide recorded).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const, data_loader as dl, physics
from core import val_metrics as vm

SATURATION_GRID = np.round(np.arange(0.60, 1.00, 0.01), 3)
EVENT_WINDOW_DAYS = 2
RAIN_START_YEAR = 1991
TRAIN_FRACTION = 0.6
SEED = 42
OUTDIR = Path("output/temporal/01_calibrate_saturation")
OUTDIR.mkdir(parents=True, exist_ok=True)

SCREENING_LOADER = dl.load_wsl_inventory


def load_locked_onset():
    """Read the onset selected by 01_sweep_onset.py, instead of relying on
    constants.py being kept in sync by hand."""
    path = Path("output/temporal/01_sweep_onset/onset_summary.csv")
    if not path.exists():
        raise FileNotFoundError("Run 01_sweep_onset.py first.")
    summary = pd.read_csv(path)
    return float(summary.loc[summary["youden_j"].idxmax(), "onset"])


S_ONSET = load_locked_onset()


def load_train_events():
    """Reuse the split from 01_sweep_onset.py so both scripts compare the
    exact same events."""
    path = Path("output/temporal/01_sweep_onset/train_events.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Run 01_sweep_onset.py first to create the train/test split."
        )
    return pd.read_csv(path, parse_dates=["date"])


def peak_saturation(row, control_date):
    """Highest simulated saturation near the event date and one year earlier."""
    years_needed = {row["date"].year, control_date.year}
    start_year = max(min(years_needed) - 1, RAIN_START_YEAR)
    end_year = max(years_needed) + 1
    rain = dl.load_rainfall(row["x"], row["y"], range(start_year, end_year))
    region = dl.assign_region(row["x"], row["y"])
    if rain is None or rain.empty or region is None:
        return np.nan, np.nan
    drainage, et = dl.load_calibration_params(region)
    if drainage is None:
        return np.nan, np.nan

    saturation = pd.Series(
        physics.calculate_daily_saturation(
            rain.to_numpy(dtype=float),
            n=const.N,
            n_perp=const.H_PERP,
            m0=S_ONSET,
            s_pp_onset=S_ONSET,
            drainage_rate=drainage,
            et_rate=et,
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

    return peak_near(row["date"]), peak_near(control_date)


def sweep(train):
    full_inventory = SCREENING_LOADER()
    control_dates = train.apply(
        lambda row: vm.find_control_date(
            row["x"],
            row["y"],
            row["date"],
            full_inventory,
            rain_start_year=RAIN_START_YEAR,
        ),
        axis=1,
    )

    train = train[control_dates.notna()].copy()
    control_dates = control_dates.dropna()

    peaks = pd.DataFrame(
        [
            peak_saturation(row, control_date)
            for (_, row), control_date in zip(train.iterrows(), control_dates)
        ],
        columns=["event_saturation", "control_saturation"],
    )
    peaks = peaks.dropna()
    print(f"{len(peaks)}/{len(train)} events have usable rainfall + calibration data")

    rows = []
    for s_crit in SATURATION_GRID:
        event_fail = peaks["event_saturation"] >= s_crit
        control_fail = peaks["control_saturation"] >= s_crit
        rows.append(
            {
                "s_crit": s_crit,
                "youden_j": vm.youden_j(event_fail, control_fail),
                "mcc": vm.mcc(event_fail, control_fail),
                "auc": vm.pairwise_auc(
                    peaks["control_saturation"], peaks["event_saturation"]
                ),
                "event_hit_rate": event_fail.mean(),
                "control_failure_rate": control_fail.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot(summary):

    best = summary.loc[summary["youden_j"].idxmax()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["s_crit"], summary["youden_j"], "o-", label="Youden's J")
    ax.plot(summary["s_crit"], summary["mcc"], "s-", label="MCC")
    ax.plot(summary["s_crit"], summary["auc"], "^-", label="AUC")
    ax.axvline(
        best["s_crit"],
        ls="--",
        color="black",
        label=f"Best threshold = {best['s_crit']:.2f}",
    )
    ax.set_xlabel("Saturation threshold")
    ax.set_ylabel("Metric value (Youden's J, MCC, AUC)  \n")
    ax.set_title("Which saturation level best predicts a landslide?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "saturation_sweep.png", dpi=150)
    plt.close(fig)


def main():
    train = load_train_events()
    summary = sweep(train)
    summary.to_csv(OUTDIR / "saturation_summary.csv", index=False)
    plot(summary)

    best = summary.loc[summary["youden_j"].idxmax()]
    print(
        f"Best saturation threshold: {best['s_crit']:.2f}  "
        f"(Youden's J = {best['youden_j']:.3f}, MCC = {best['mcc']:.3f})"
    )

    locked = {
        "s_onset": S_ONSET,
        "s_crit": float(best["s_crit"]),
        "youden_j": float(best["youden_j"]),
        "mcc": float(best["mcc"]),
    }
    with open(OUTDIR / "locked_params.json", "w", encoding="utf-8") as f:
        json.dump(locked, f, indent=2)
    print(f"Locked parameters written to {OUTDIR / 'locked_params.json'}")


if __name__ == "__main__":
    main()
