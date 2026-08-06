"""01_sweep_onset.py

Sweeps the pore-pressure onset threshold and checks: at which value does the
model best separate days with a recorded landslide from the same location
one year earlier (no landslide)?

Metric: Youden's J = TPR - FPR at FoS <= 1. Not AUC - AUC ranks by pore
pressure and is mathematically blind to onset (see calibrate_saturation.py).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const, data_loader as dl, physics
from core import val_metrics as vm

TRAIN_FRACTION = 0.6
SEED = 42
EVENT_WINDOW_DAYS = 2
RAIN_START_YEAR = 1991
BACKGROUND_SATURATION = 0.60  # bucket starting state
ONSET_GRID = np.round(np.arange(0.50, 0.90, 0.02), 3)
OUTDIR = Path("output/temporal/01_sweep_onset")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASET_LOADER = dl.load_wsl_usable_inventory
SCREENING_LOADER = dl.load_wsl_usable_inventory


def load_train_events():
    """WSL events used to pick the onset (40% held out for final testing)."""
    path = OUTDIR / "train_events.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    events = DATASET_LOADER()
    train = events.sample(frac=TRAIN_FRACTION, random_state=SEED)
    events.drop(train.index).to_csv(OUTDIR / "test_events.csv", index=False)
    train.to_csv(path, index=False)
    return train


def load_rain_and_params(row, control_date):
    """Rainfall series + drainage/ET for one event, or None if unavailable."""
    years_needed = {row["date"].year, control_date.year}
    start_year = max(min(years_needed) - 1, RAIN_START_YEAR)
    end_year = max(years_needed) + 1
    rain = dl.load_rainfall(row["x"], row["y"], range(start_year, end_year))
    region = dl.assign_region(row["x"], row["y"])
    if rain is None or rain.empty or region is None:
        return None
    drainage, et, et_amp = dl.load_calibration_params(region)
    return None if drainage is None else (rain, drainage, et, et_amp)


def min_fos_near(rain, drainage, et, et_amp, onset, date):
    """Lowest Factor of Safety within EVENT_WINDOW_DAYS of `date`."""
    saturation = physics.calculate_daily_saturation(
        rain.to_numpy(dtype=float),
        n=const.N,
        n_perp=const.H_PERP,
        m0=BACKGROUND_SATURATION,
        s_pp_onset=onset,
        drainage_rate=drainage,
        et_rate=et,
        day_of_year=rain.index.dayofyear.to_numpy(),
        et_amplitude=et_amp,
    )
    fos = physics.compute_fos(
        m_array=physics.pore_pressure_ratio(saturation, onset),
        c=const.C,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )
    fos = pd.Series(fos, index=rain.index)
    window = fos.loc[
        date
        - pd.Timedelta(days=EVENT_WINDOW_DAYS) : date
        + pd.Timedelta(days=EVENT_WINDOW_DAYS)
    ]
    return float(window.min()) if not window.empty else np.nan


def sweep(train):
    """Load rainfall once per event, then score every onset value cheaply."""
    full_inventory = SCREENING_LOADER()
    prepared = []
    dropped_no_control = dropped_no_rain = 0
    for _, row in train.iterrows():
        control_date = vm.find_control_date(
            row["x"],
            row["y"],
            row["date"],
            full_inventory,
            rain_start_year=RAIN_START_YEAR,
        )
        if control_date is None:
            dropped_no_control += 1
            continue
        loaded = load_rain_and_params(row, control_date)
        if loaded is None:
            dropped_no_rain += 1
            continue
        prepared.append((row["date"], control_date, *loaded))

    print(
        f"{len(prepared)}/{len(train)} events have usable rainfall + calibration data "
        f"({dropped_no_control} dropped - no clean control date, "
        f"{dropped_no_rain} dropped - no rainfall/calibration data)"
    )

    rows = []
    for onset in ONSET_GRID:
        event_fos = [
            min_fos_near(rain, d, et, a, onset, date)
            for date, _, rain, d, et, a in prepared
        ]
        control_fos = [
            min_fos_near(rain, d, et, a, onset, control_date)
            for _, control_date, rain, d, et, a in prepared
        ]
        event_fail = np.array(event_fos) <= 1.0
        control_fail = np.array(control_fos) <= 1.0
        rows.append(
            {
                "onset": onset,
                "youden_j": vm.youden_j(event_fail, control_fail),
                "mcc": vm.mcc(event_fail, control_fail),
                "auc": vm.pairwise_auc(event_fos, control_fos),
                "event_hit_rate": event_fail.mean(),
                "control_failure_rate": control_fail.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot(summary):

    best = summary.loc[summary["youden_j"].idxmax()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["onset"], summary["youden_j"], "o-", label="Youden's J")
    ax.plot(summary["onset"], summary["mcc"], "s-", label="MCC")
    ax.plot(summary["onset"], summary["auc"], "^-", label="AUC")
    ax.axvline(
        best["onset"], ls="--", color="black", label=f"Best onset = {best['onset']:.2f}"
    )
    ax.set_xlabel("Onset threshold")
    ax.set_ylabel("Metric values")
    ax.set_title(
        "Does the onset threshold predict landslides better at some values than others?"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "onset_sweep.png", dpi=150)
    plt.close(fig)


def main():
    train = load_train_events()
    summary = sweep(train)
    summary.to_csv(OUTDIR / "onset_summary.csv", index=False)
    plot(summary)

    best = summary.loc[summary["youden_j"].idxmax()]
    print(
        f"Best onset: {best['onset']:.2f}  "
        f"(Youden's J = {best['youden_j']:.3f}, MCC = {best['mcc']:.3f})"
    )


if __name__ == "__main__":
    main()
