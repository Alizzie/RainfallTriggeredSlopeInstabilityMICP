import os
import numpy as np
import pandas as pd
import xarray as xr

import bucket_model as bm
import model as mod
import constants as const

VALIDATION_REGION = 65
YEARS = range(2005, 2016)

# Event window instead of a single date: the recorded failure is 16 Nov 2014,
# but peak rainfall is 15 Nov and inventory dates carry ~1-day uncertainty.
# We report the MINIMUM FoS across this window.
EVENT_WINDOW = (pd.Timestamp("2014-11-14"), pd.Timestamp("2014-11-17"))
RECORDED_FAILURE = pd.Timestamp("2014-11-16")  # kept only for reporting

DRY_DAY_MM = 5.0
BACKGROUND_TARGET = 0.60


def get_average_coordinates(region_id):
    path = f"data/trockenheit_grenzcoord/data_region{region_id - 30}.csv"
    c = pd.read_csv(path, sep=";")
    return c["Easting"].mean(), c["Northing"].mean()


def load_rainfall(region_id, years):
    e, n = get_average_coordinates(region_id)
    frames = []
    for yr in years:
        fp = f"data/rhiresD/ogd-surface-derived-grid-archive.rhiresd_ch01h.swiss.lv95_{yr}0101000000_{yr}1231000000.nc"
        if not os.path.exists(fp):
            continue
        ds = xr.open_dataset(fp)
        r = ds["RhiresD"].sel(E=e, N=n, method="nearest")
        frames.append(
            pd.Series(
                np.nan_to_num(r.values, nan=0.0), index=pd.to_datetime(r.time.values)
            )
        )
    return pd.concat(frames).sort_index()


def to_effective(S):
    return np.clip((S - const.S_RES) / (1.0 - const.S_RES), 0.0, 1.0)


def load_drainage_et(region_id):
    row = (
        pd.read_csv("output/calibration_results.csv")
        .query("region_id == @region_id")
        .iloc[0]
    )
    return float(row["drainage"]), float(row["et"])


def main():
    rain = load_rainfall(VALIDATION_REGION, YEARS)
    drainage, et = load_drainage_et(VALIDATION_REGION)
    dry_mask = rain.values < DRY_DAY_MM

    print(f"Region {VALIDATION_REGION} | drainage={drainage:.3f} et={et:.3f}")
    print(
        f"Recorded failure: {RECORDED_FAILURE.date()} | "
        f"window {EVENT_WINDOW[0].date()}–{EVENT_WINDOW[1].date()}"
    )
    print(f"{'onset':>7} {'bg_Se':>8} {'min_FoS':>11} {'on_date':>12} {'bg_err':>8}")

    rows = {}
    for onset in const.S_PP_ONSET_SWEEP:
        S = bm.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=0.60,
            s_pp_onset=onset,
            drainage_rate=drainage,
            et_rate=et,
        )
        S = pd.Series(S, index=rain.index)
        Se = to_effective(S)

        bg = float(np.median(Se[dry_mask]))

        m_pp = pd.Series(bm.pore_pressure_ratio(S.values, onset), index=S.index)
        fos_series = pd.Series(
            mod.compute_fos(
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

    print(
        f"\nClosest to background {BACKGROUND_TARGET}: onset "
        f"{min(rows, key=lambda o: rows[o]['bg_err']):.2f}"
    )
    print(
        "Pick by hand: min_FoS ≤ 1 in the window is the failure check; "
        "bg_Se near 0.60 is the sanity anchor. Set S_PP_ONSET_DEFAULT in constants.py."
    )


if __name__ == "__main__":
    main()
