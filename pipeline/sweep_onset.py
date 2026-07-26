"""
Sweep the S_PP_ONSET parameter to find a value that produces a reasonable background saturation (Se) and also predicts failure during the Bondo landslide event in November 2014.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import physics
from core import constants as const

VALIDATION_REGION = 65
YEARS = range(2005, 2016)
EVENT_WINDOW = (pd.Timestamp("2014-11-14"), pd.Timestamp("2014-11-17"))
RECORDED_FAILURE = pd.Timestamp("2014-11-16")  # kept only for reporting

DRY_DAY_MM = 5.0
BACKGROUND_TARGET = 0.60


def to_effective(S):
    return np.clip((S - const.S_RES) / (1.0 - const.S_RES), 0.0, 1.0)


def main():
    easting, norting = dl.get_region_coordinates(VALIDATION_REGION)
    rain = dl.load_rainfall(easting, norting, YEARS)
    drainage, et = dl.load_calibration_params(VALIDATION_REGION)

    if rain is None or drainage is None:
        print("Error: Missing data or calibration parameters.")
        return

    dry_mask = rain.values < DRY_DAY_MM

    print(f"Region {VALIDATION_REGION} | drainage={drainage:.3f} et={et:.3f}")
    print(
        f"Recorded failure: {RECORDED_FAILURE.date()} | "
        f"window {EVENT_WINDOW[0].date()}–{EVENT_WINDOW[1].date()}"
    )
    print(f"{'onset':>7} {'bg_Se':>8} {'min_FoS':>11} {'on_date':>12} {'bg_err':>8}")

    rows = {}
    for onset in const.S_PP_ONSET_SWEEP:
        S = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=BACKGROUND_TARGET,
            s_pp_onset=onset,
            drainage_rate=drainage,
            et_rate=et,
        )

        S = pd.Series(S, index=rain.index)
        Se = to_effective(S)

        bg = float(np.median(Se[dry_mask]))
        m_pp = pd.Series(physics.pore_pressure_ratio(S.values, onset), index=S.index)

        fos_series = pd.Series(
            physics.compute_fos(
                m_array=m_pp.values,
                c=const.C,
                gamma=const.GAMMA,
                gamma_w=const.GAMMA_W,
                h_v=const.H_V,
                beta_rad=const.beta,
                phi_rad=const.phi,
            ),
            index=S.index,
        )

        win = fos_series.loc[EVENT_WINDOW[0] : EVENT_WINDOW[1]]
        fos_min = float(win.min())
        fos_min_date = win.idxmin().date()

        bg_err = abs(bg - BACKGROUND_TARGET)
        rows[onset] = {"bg_err": bg_err}
        print(
            f"{onset:7.2f} {bg:8.3f} {fos_min:11.2f}  {str(fos_min_date):>10}  {bg_err:7.3f}"
        )

    best_onset = min(rows, key=lambda o: rows[o]["bg_err"])
    print(f"\nClosest to background {BACKGROUND_TARGET}: onset {best_onset:.2f}")


if __name__ == "__main__":
    main()
