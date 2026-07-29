"""
02_calibrate_model.py - Regional Drainage and ET Calibration

This script calibrates the hydrological parameters (drainage and evapotranspiration rates)
for all 38 Swiss drought regions. It utilizes SciPy's minimization algorithm to find
the rates that produce simulated soil moisture levels most closely matching the historical
BAFU field capacity (nFK) observations.

Optimization Note: The objective function relies entirely on vectorized NumPy arrays
to ensure rapid convergence across the spatial datasets.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import physics
from core import constants as const

REGION_IDS = range(31, 69)  # Drought region IDs from 31 to 68
YEARS = range(1991, 2026)
OUTPUT_DIR = "output/02_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def objective(params, rf_values, nfk_common, nfk0, common_idx):
    """
    Highly optimized objective function for SciPy.
    Calculates the Mean Squared Error (MSE) between the simulated bucket
    saturation and the observed BAFU nFK values.
    """
    d_rate, et_rate = params

    # 1. Run physical bucket simulation
    sim_array = physics.calculate_daily_saturation(
        rf_values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=nfk0 * const.S_PP_ONSET_DEFAULT,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,
    )

    # 3. Convert absolute saturation to field capacity representation and clip
    pred_band = np.clip(sim_array[common_idx] / const.S_PP_ONSET_DEFAULT, 0.0, 1.0)

    # 4. Return Mean Squared Error
    return np.mean((pred_band - nfk_common) ** 2)


def main():
    """Main Execution"""
    results = []
    for rid in REGION_IDS:
        fits = []

        try:
            avg_e, avg_n = dl.get_region_coordinates(rid)
        except Exception as e:
            print(f"LOG: skipping region {rid} (error loading coordinates): {e}")
            continue

        print(f"Calibrating Region {rid}")

        for yr in YEARS:
            rf = dl.load_rainfall(avg_e, avg_n, yr)
            nfk = dl.load_bafu_moisture(rid, yr, interpolate_daily=True)

            if rf is None or len(nfk) < 5:
                print(f"LOG: skipping region {rid}, year {yr} (insufficient data).")
                continue

            # 1. Check for overlapping dates between rainfall and nFK data
            common_idx = np.where(np.isin(rf.index, nfk.index))[0]
            if len(common_idx) == 0:
                continue  # No overlapping dates, skip this year

            common_dates = rf.index[common_idx]
            nfk_common = nfk.reindex(common_dates).to_numpy()

            # 2. Run minimization to find optimal [drainage, ET] rates
            res = minimize(
                objective,
                x0=[0.2, 1.5],
                args=(rf.to_numpy(), nfk_common, nfk.iloc[0], common_idx),
                bounds=[(0.01, 0.5), (0.0, 5.0)],
            )
            fits.append(res.x)

        if fits:
            # Average the optimized rates across all valid years for the region
            avg = np.mean(fits, axis=0)
            results.append(
                {
                    "region_id": rid,
                    "easting": avg_e,
                    "northing": avg_n,
                    "drainage": avg[0],
                    "et": avg[1],
                }
            )

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(results).to_csv(dl.PATH_CALIB, index=False)
    print(f"Calibration complete -> {dl.PATH_CALIB}.")


if __name__ == "__main__":
    main()
