"""
01_sweep_onset.py - Pore-Pressure Activation Threshold Optimizer

This script iteratively sweeps through potential pore-pressure onset thresholds (S_PP_ONSET).
It aims to find a threshold that maintains a stable, realistic background soil moisture (Se)
during dry periods, while successfully predicting slope failure (FoS <= 1.0) during a known
historical extreme precipitation event.

Target Event: The August 2005 Alpine Flood and Landslide Event (Central Switzerland).
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import physics
from core import constants as const
from validation import val_visuals as vis

# Target Region: Central Switzerland
VALIDATION_REGION = [33, 38, 40]
YEARS = range(2000, 2010)

# Target Event: August 2005 Swiss Storm
EVENT_WINDOW = (pd.Timestamp("2005-08-20"), pd.Timestamp("2005-08-25"))
RECORDED_FAILURE = pd.Timestamp("2005-08-22")

DRY_DAY_MM = 5.0
BACKGROUND_TARGET = 0.60
PLOT_DIR = "output/01_sweep_onset"


def to_effective(S):
    """Converts absolute saturation to effective saturation (Se) considering residual water."""
    return np.clip((S - const.S_RES) / (1.0 - const.S_RES), 0.0, 1.0)


def main():
    """Main function to sweep through S_PP_ONSET values and evaluate model performance."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    onset_values = []
    bg_error = []

    for region in VALIDATION_REGION:
        easting, norting = dl.get_region_coordinates(region)
        rain = dl.load_rainfall(easting, norting, YEARS)
        drainage, et = dl.load_calibration_params(region)

        if rain is None or drainage is None:
            print("Error: Missing data or calibration parameters.")
            continue

        # Filter for days with minimal rainfall to establish background moisture state
        dry_mask = rain.values < DRY_DAY_MM

        print(
            f"\nRegion {region} Calibration | Drainage={drainage:.3f} mm/d | ET={et:.3f} mm/d"
        )
        print(
            f"Target Event: {RECORDED_FAILURE.date()} | Evaluation Window: {EVENT_WINDOW[0].date()} to {EVENT_WINDOW[1].date()}"
        )
        print("-" * 75)
        print(
            f"{'Onset':>7} {'Bg_Se':>8} {'Min_FoS':>11} {'Failure_Date':>14} {'Bg_Error':>10}"
        )

        results = {}

        # Sweep through the potential onset thresholds defined in constants
        for onset in const.S_PP_ONSET_SWEEP:
            # 1. Simulate the daily saturation using the discrete bucket model
            S = physics.calculate_daily_saturation(
                rain.values,
                n=const.N,
                n_perp=const.H_PERP,
                m0=BACKGROUND_TARGET,
                s_pp_onset=onset,
                drainage_rate=drainage,
                et_rate=et,
            )

            S_series = pd.Series(S, index=rain.index)
            Se = to_effective(S)

            # 2. Calculate the median background effective saturation during dry days
            bg = float(np.median(Se[dry_mask]))

            # 3. Calculate dynamic pore pressure and FoS for the entire simulation period
            m_pp = pd.Series(
                physics.pore_pressure_ratio(S_series.values, onset),
                index=S_series.index,
            )
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
                index=S_series.index,
            )

            # 4. Evaluate FoS during the specific historical event window
            win = fos_series.loc[EVENT_WINDOW[0] : EVENT_WINDOW[1]]
            fos_min = float(win.min())
            fos_min_date = win.idxmin().date()

            # 5. Track the error relative to our realistic baseline target
            bg_err = abs(bg - BACKGROUND_TARGET)
            results[onset] = {"bg_err": bg_err}
            print(
                f"{onset:7.2f} {bg:8.3f} {fos_min:11.2f}  {str(fos_min_date):>10}  {bg_err:7.3f}"
            )

        # Determine best onset
        best_onset = min(results, key=lambda o: results[o]["bg_err"])
        print("-" * 75)
        print(
            f"Optimal Onset Threshold: {best_onset:.2f} (Closest to background Se of {BACKGROUND_TARGET})"
        )

        onset_values.append(best_onset)
        bg_error.append(results[best_onset]["bg_err"])

        # ==========================================================
        # Generate Plot for the Best Onset
        # ==========================================================
        # Re-run simulation strictly for the winning onset to get correct arrays
        S_best = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=BACKGROUND_TARGET,
            s_pp_onset=best_onset,
            drainage_rate=drainage,
            et_rate=et,
        )
        S_series_best = pd.Series(S_best, index=rain.index)
        m_pp_best = pd.Series(
            physics.pore_pressure_ratio(S_series_best.values, best_onset),
            index=rain.index,
        )
        fos_series_best = pd.Series(
            physics.compute_fos(
                m_array=m_pp_best.values,
                c=const.C,
                gamma=const.GAMMA,
                gamma_w=const.GAMMA_W,
                h_v=const.H_V,
                beta_rad=const.beta,
                phi_rad=const.phi,
            ),
            index=rain.index,
        )

        bafu = dl.load_bafu_moisture(region, interpolate_daily=False)

        # Define a clean 30-day window for the plot so it is highly readable
        plot_start = RECORDED_FAILURE - pd.Timedelta(days=15)
        plot_end = RECORDED_FAILURE + pd.Timedelta(days=15)

        vis.plot_event(
            rain=rain.loc[plot_start:plot_end],
            S=S_series_best.loc[plot_start:plot_end],
            fos=fos_series_best.loc[plot_start:plot_end],
            bafu=(
                bafu.loc[plot_start:plot_end]
                if bafu is not None
                else pd.Series(dtype=float)
            ),
            date=RECORDED_FAILURE,
            name=f"Region {region} (August 2005 Storm)",
            idx=region,
            plot_dir=PLOT_DIR,
            onset_val=best_onset,
            c_val=const.C,
        )
        print(f"--> Saved plot for Region {region} to {PLOT_DIR}/")

    print("\nSummary of Optimal Onset Thresholds for Validation Regions:")
    print(f"Best Onset Value Mean: {np.mean(onset_values):.3f}")
    print(f"Avg Background Se Error: {np.mean(bg_error):.3f}")


if __name__ == "__main__":
    main()
