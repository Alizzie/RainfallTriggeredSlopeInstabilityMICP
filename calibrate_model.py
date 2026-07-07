import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
import bucket_model as bm
import constants as const

REGION_IDS = range(31, 69)  # Drought region IDs from 31 to 68
YEARS = range(1991, 2026)
LAMBDA = 1.0


def get_average_coordinates(region_id):
    path = f"data/trockenheit_grenzcoord/data_region{region_id - 30}.csv"
    region_coords = pd.read_csv(path, sep=";")
    return region_coords["Easting"].mean(), region_coords["Northing"].mean()


def load_and_prepare_data(region_id, year):
    # 1. Load Rainfall
    file_path = f"data/rhiresD/ogd-surface-derived-grid-archive.rhiresd_ch01h.swiss.lv95_{year}0101000000_{year}1231000000.nc"
    if not os.path.exists(file_path):
        return None, None

    ds = xr.open_dataset(file_path)
    avg_e, avg_n = get_average_coordinates(region_id)
    rainfall = np.nan_to_num(
        ds["RhiresD"].sel(E=avg_e, N=avg_n, method="nearest").values, nan=0.0
    )

    # 2. Load BAFU and Interpolate to Daily
    bafu = pd.read_csv(
        "data/soil_moisture_history/weekly_historic_regions.csv",
        sep=",",
        skiprows=3,
        parse_dates=["measured_at"],
        dayfirst=True,
    )
    bafu_reg = bafu[bafu["drought_region_id"] == region_id].copy()
    bafu_reg = bafu_reg[bafu_reg["measured_at"].dt.year == year].set_index(
        "measured_at"
    )

    if len(bafu_reg) < 5:
        return None, None  # Skip if insufficient data

    rainfall_series = pd.Series(
        rainfall, index=pd.date_range(start=f"{year}-01-01", periods=len(rainfall))
    )

    # 1. Perform the interpolation
    nfk = (
        bafu_reg["soil_moisture_ufc"].resample("D").interpolate(method="linear") / 100.0
    )

    return rainfall_series, nfk


def objective(params, rainfall, nfk):
    d_rate, et_rate = params

    # 1. Run simulation (result is a numpy array)
    sim_array = bm.calculate_daily_saturation(
        rainfall.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=nfk.iloc[0] * const.S_PP_ONSET_DEFAULT,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,
    )

    sim_array = pd.Series(sim_array, index=rainfall.index)

    pred_band = (sim_array / const.S_PP_ONSET_DEFAULT).clip(0, 1)
    m_pp = pd.Series(
        bm.pore_pressure_ratio(sim_array.values, const.S_PP_ONSET_DEFAULT),
        index=sim_array.index,
    )

    common = nfk.index.intersection(sim_array.index)
    b = nfk.loc[common]

    shape_err = ((pred_band.loc[common] - b) ** 2).mean()
    sub_onset = b < 0.999
    drain_err = (m_pp.loc[common][sub_onset] ** 2).mean() if sub_onset.any() else 0.0
    return shape_err + LAMBDA * drain_err


# --- EXECUTION ---
results = []
for rid in REGION_IDS:
    fits = []

    try:
        avg_e, avg_n = get_average_coordinates(rid)
    except Exception as e:
        print(f"LOG: skipping region {rid} (error loading coordinates): {e}")
        continue

    for yr in YEARS:
        rf, nfk = load_and_prepare_data(rid, yr)
        if rf is None or len(nfk) < 5:
            print(f"LOG: skipping region {rid}, year {yr} (insufficient data).")
            continue
        res = minimize(
            objective, x0=[0.2, 1.5], args=(rf, nfk), bounds=[(0.01, 0.5), (0.0, 5.0)]
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

pd.DataFrame(results).to_csv("output/calibration_results.csv", index=False)
print("Calibration complete -> 'output/calibration_results.csv'.")
