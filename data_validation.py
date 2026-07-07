# batch_validate.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import constants as const
import model as mod
import bucket_model as bm

INVENTORY_CSV = "data/wsl_landslide.csv"  # your sheet
PLOT_DIR = "output/hist_plots"
RESULTS_CSV = "output/validation_results.csv"
CALIBRATION_CSV = "output/calibration_results.csv"
WINDOW_DAYS = 2  # ±2 days around the recorded date
SPINUP_DAYS = 120  # antecedent period for the bucket
FOS_THRESHOLD = 1.0

os.makedirs(PLOT_DIR, exist_ok=True)


def get_calibrated_params(x, y):
    if os.path.exists(CALIBRATION_CSV):
        calib = pd.read_csv(CALIBRATION_CSV)
        # find the closest region by Euclidean distance
        calib["dist"] = np.sqrt(
            (calib["easting"] - x) ** 2 + (calib["northing"] - y) ** 2
        )
        closest = calib.loc[calib["dist"].idxmin()]
        return (
            int(closest["region_id"]),
            float(closest["drainage"]),
            float(closest["et"]),
        )
    else:
        print(
            f"Warning: Calibration file '{CALIBRATION_CSV}' not found. Using default drainage and ET rates."
        )
        return None, 0.5, 2.0


def to_lv95(x, y):
    """Inventory looks like LV03 (6-digit). NetCDF grid is LV95 (7-digit)."""
    if x < 1_000_000:  # LV03 easting ~600000-800000
        x += 2_000_000
    if y < 1_000_000:  # LV03 northing ~100000-300000
        y += 1_000_000
    return x, y


def load_inventory(path):
    df = pd.read_csv(path, skiprows=3)
    # adjust these names to your actual header row
    df = df.rename(
        columns={
            "x-coordinate": "x",
            "y-coordinate": "y",
            "date": "date",
            "name of municipality": "municipality",
        }
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna(subset=["x", "y"]).reset_index(drop=True)


def load_rainfall_at(x, y, year):
    fp = (
        f"data/rhiresD/ogd-surface-derived-grid-archive.rhiresd_ch01h."
        f"swiss.lv95_{year}0101000000_{year}1231000000.nc"
    )
    if not os.path.exists(fp):
        return None
    ds = xr.open_dataset(fp)
    r = ds["RhiresD"].sel(E=x, N=y, method="nearest")
    return pd.Series(
        np.nan_to_num(r.values, nan=0.0), index=pd.to_datetime(r.time.values)
    )


def simulate_event(x, y, date):
    x, y = to_lv95(x, y)
    region_id, d_rate, et_rate = get_calibrated_params(x, y)

    start = date - pd.Timedelta(days=SPINUP_DAYS)
    end = date + pd.Timedelta(days=WINDOW_DAYS + 5)

    # rainfall may span a year boundary; load each needed year
    years = sorted({start.year, end.year})
    parts = [load_rainfall_at(x, y, yr) for yr in years]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    rain = pd.concat(parts).sort_index()
    rain = rain.loc[start:end]
    if rain.empty:
        return None

    S = bm.calculate_daily_saturation(
        rain.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=0.60,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=d_rate,
        et_rate=et_rate,  # or per-region from calibration
    )
    S = pd.Series(S, index=rain.index)
    m_pp = bm.pore_pressure_ratio(S.values, const.S_PP_ONSET_DEFAULT)
    fos = pd.Series(
        mod.compute_fos(
            m_array=m_pp,
            c=const.C,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=const.H_V,
            beta_rad=const.beta,
            phi_rad=const.phi,
        ),
        index=S.index,
    )
    return rain, S, fos, region_id, d_rate, et_rate


def evaluate(fos, date):
    win = fos.loc[
        date - pd.Timedelta(days=WINDOW_DAYS) : date + pd.Timedelta(days=WINDOW_DAYS)
    ]
    if win.empty:
        return None, None
    return float(win.min()), win.idxmin().date()


def plot_event(rain, S, fos, date, name, idx):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    a1.bar(rain.index, rain.values, color="blue", alpha=0.6)
    a1.set_ylabel("Rain (mm/day)")
    a1.set_title(f"{name} — {date.date()}")
    a2.plot(S.index, S.values, color="purple")
    a2.axhline(const.S_PP_ONSET_DEFAULT, color="orange", ls=":")
    a2.set_ylabel("Saturation")
    a2.set_ylim(0, 1.1)
    a3.plot(fos.index, fos.values, color="red")
    a3.axhline(1.0, color="gray", ls="-.")
    a3.axvline(date, color="black", ls="--", alpha=0.5)
    a3.set_ylabel("FoS")
    a3.set_ylim(0.5, 4.5)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/event_{idx:03d}_{date.date()}.png", dpi=150)
    plt.close(fig)


def main():
    inv = load_inventory(INVENTORY_CSV)
    print(f"{len(inv)} events with usable date + coordinates.")

    rows = []
    for idx, ev in inv.iterrows():
        out = simulate_event(ev["x"], ev["y"], ev["date"])
        if out is None:
            rows.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue
        rain, S, fos, region_id, d_rate, et_rate = out
        min_fos, min_date = evaluate(fos, ev["date"])
        if min_fos is None:
            rows.append({**ev, "min_fos": np.nan, "label": "no_data"})
            continue
        label = "unstable" if min_fos <= FOS_THRESHOLD else "stable"
        plot_event(rain, S, fos, ev["date"], ev.get("municipality", "event"), idx)
        rows.append(
            {
                "municipality": ev.get("municipality", ""),
                "date": ev["date"].date(),
                "x": ev["x"],
                "y": ev["y"],
                "min_fos": round(min_fos, 3),
                "min_fos_date": min_date,
                "rain_max_window": round(
                    rain.loc[
                        ev["date"]
                        - pd.Timedelta(days=WINDOW_DAYS) : ev["date"]
                        + pd.Timedelta(days=WINDOW_DAYS)
                    ].max(),
                    1,
                ),
                "label": label,
                "region_id": region_id,
                "et_rate": et_rate,
                "drainage_rate": d_rate,
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_CSV, index=False)

    scored = res[res["label"].isin(["stable", "unstable"])]
    print("stable events, rain in window:")
    print(scored[scored.label == "stable"]["rain_max_window"].describe())
    print("\nunstable events, rain in window:")
    print(scored[scored.label == "unstable"]["rain_max_window"].describe())
    print(
        "\nstable events with <5mm rain in window:",
        (scored[scored.label == "stable"]["rain_max_window"] < 5).sum(),
    )
    n = len(scored)
    detected = (scored["label"] == "unstable").sum()
    print(
        f"\nDetection rate: {detected}/{n} = {detected/n:.1%} "
        f"of events had FoS ≤ 1 within ±{WINDOW_DAYS} days."
    )
    print(f"Results -> {RESULTS_CSV} | plots -> {PLOT_DIR}/")

    for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 999)]:
        band = scored[(scored.rain_max_window >= lo) & (scored.rain_max_window < hi)]
        if len(band):
            det = (band.label == "unstable").mean()
            print(f"{lo:3d}-{hi:3d} mm: {det:.0%} detected  (n={len(band)})")

    # summary: min-FoS distribution across events (the event-side of a future AUC)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scored["min_fos"], bins=20, color="steelblue", edgecolor="k")
    ax.axvline(1.0, color="red", ls="--", label="Failure threshold")
    ax.set_xlabel("Minimum FoS in event window")
    ax.set_ylabel("Events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/_summary_minfos_hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

# Without calibration
#   0- 20 mm: 0% detected  (n=175)
#  20- 40 mm: 11% detected  (n=186)
#  40- 80 mm: 65% detected  (n=162)
#  80-999 mm: 98% detected  (n=115)


# With calibration
# stable events, rain in window:
# count    392.000000
# mean      23.577041
# std       15.255474
# min        0.000000
# 25%       12.800000
# 50%       22.150000
# 75%       32.700001
# max       72.500000
# Name: rain_max_window, dtype: float64

# unstable events, rain in window:
# count    246.000000
# mean      83.780488
# std       41.770840
# min       31.200001
# 25%       54.450001
# 50%       78.350002
# 75%      102.450003
# max      283.600006
# Name: rain_max_window, dtype: float64

# stable events with <5mm rain in window: 45

# Detection rate: 246/638 = 38.6% of events had FoS ≤ 1 within ±2 days.
# Results -> output/validation_results.csv | plots -> output/hist_plots/
#   0- 20 mm: 0% detected  (n=175)
#  20- 40 mm: 13% detected  (n=186)
#  40- 80 mm: 66% detected  (n=162)
#  80-999 mm: 100% detected  (n=115)
