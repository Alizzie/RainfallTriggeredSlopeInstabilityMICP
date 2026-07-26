"""
Calibrate the bucket model parameters (drainage and ET rates) for each drought region using historical rainfall and soil moisture data.
This script iterates over all drought regions and years,
loading the corresponding rainfall and BAFU soil moisture data,
and optimizes the drainage and ET rates to minimize the error between the simulated bucket saturation and the observed BAFU nFK values.
The results are saved to a CSV file for further analysis and visualization.
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
LAMBDA = 1


def objective(params, rainfall, nfk):
    """Objective function to minimize: the error between simulated and observed nFK values."""
    d_rate, et_rate = params

    # 1. Run simulation (result = simulated daily saturation)
    sim_array = physics.calculate_daily_saturation(
        rainfall.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=nfk.iloc[0] * const.S_PP_ONSET_DEFAULT,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,
    )

    sim_array = pd.Series(sim_array, index=rainfall.index)

    # pred_band = how full the bucket is relative to the onset threshold, clipped to [0, 1]
    pred_band = (sim_array / const.S_PP_ONSET_DEFAULT).clip(0, 1)

    # Days where both simulation and nFK data are available
    common = nfk.index.intersection(sim_array.index)

    # b = observed nFK values for the common days
    b = nfk.loc[common]

    # m_pp = predicted pore pressure ratio for the common days
    m_pp = pd.Series(
        physics.pore_pressure_ratio(sim_array.values, const.S_PP_ONSET_DEFAULT),
        index=sim_array.index,
    )
    # Quadratic error between predicted and observed
    shape_err = ((pred_band.loc[common] - b) ** 2).mean()
    sub_onset = b < 1
    drain_err = (m_pp.loc[common][sub_onset] ** 2).mean() if sub_onset.any() else 0.0
    return shape_err + LAMBDA * drain_err


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

        for yr in YEARS:
            rf = dl.load_rainfall(avg_e, avg_n, yr)
            nfk = dl.load_bafu_moisture(rid, yr, interpolate_daily=True)

            if rf is None or len(nfk) < 5:
                print(f"LOG: skipping region {rid}, year {yr} (insufficient data).")
                continue

            res = minimize(
                objective,
                x0=[0.2, 1.5],
                args=(rf, nfk),
                bounds=[(0.01, 0.5), (0.0, 5.0)],
            )
            fits.append(res.x)

        if fits:
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
