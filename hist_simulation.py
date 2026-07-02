import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

import constants as const
import model as mod
import bucket_model as bm

# --- 0. Limit time range ---
data = [
    {
        "start_date": pd.Timestamp("2014-11-16"),
        "end_date": pd.Timestamp("2014-11-16"),
        "gemeinde": "Colrerio",
        "region_id": 65,  # Sottoceneri
        "x_coord": 2720193,
        "y_coord": 1079228,
        "impact": "gering",
    },
    {
        "start_date": pd.Timestamp("2014-11-16"),
        "end_date": pd.Timestamp("2014-11-16"),
        "gemeinde": "Davesco-Soragno",
        "region_id": 65,  # Sottoceneri
        "x_coord": 2719090,
        "y_coord": 1099151,
        "duration": 2.5,
        "impact": "gross/katastrophal",
    },
]


def get_stable_initial_saturation(bafu_df, region_id, start_date, window=10):
    """Calculates the average saturation of the last window measurements."""
    history = bafu_df[
        (bafu_df["drought_region_id"] == region_id)
        & (bafu_df["measured_at"] < start_date)
    ].sort_values(by="measured_at", ascending=False)

    if len(history) < window:
        print(
            f"Warning: Not enough historical data for region {region_id} before {start_date}. Using available data."
        )
        return history["saturation_proxy"].iloc[0]

    return history["saturation_proxy"].head(window).mean()


def simulate(
    e_coord, n_coord, start_date: pd.Timestamp, end_date: pd.Timestamp, region_id
):
    start_date = pd.to_datetime(start_date) - pd.Timedelta(days=120)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=60)
    year = start_date.year

    # --- 1A: Load rainfall data (daily precipitation in mm/day) ---
    print("Loading MeteoSwiss NetCDF...")
    ds = xr.open_dataset(
        f"data/rhiresD/ogd-surface-derived-grid-archive.rhiresd_ch01h.swiss.lv95_{year}0101000000_{year}1231000000.nc"
    )

    precip_local = ds["RhiresD"].sel(E=e_coord, N=n_coord, method="nearest")
    precip_date = precip_local.sel(time=slice(start_date, end_date))

    rainfall_data = precip_date.values
    time_axis = precip_date.time

    rainfall_data = np.nan_to_num(rainfall_data, nan=0.0)

    # --- 1B: Load soil moisture data (for comparison) ---
    print("Loading Soil Moisture Data...")
    bafu_df = pd.read_csv(
        "data/soil_moisture_history/weekly_historic_regions.csv",
        sep=";",
        skiprows=3,
        parse_dates=["measured_at"],
        dayfirst=True,
    )

    # Convert percentage to 0 - 1.0
    bafu_df["saturation_proxy"] = bafu_df["soil_moisture_ufc"] / 100.0

    bafu_local = bafu_df[
        (bafu_df["drought_region_id"] == region_id)
        & (bafu_df["measured_at"] >= start_date)
        & (bafu_df["measured_at"] <= end_date)
    ].copy()

    # --- 1C: Load drainage and ET parameters for the region ---
    # filter by region_id
    calibration_params = pd.read_csv("output/calibration_results.csv", sep=",")
    region_params = calibration_params[calibration_params["region_id"] == region_id]
    et_rate = region_params["et"].iloc[0]
    print(et_rate)
    drainage_rate = region_params["drainage"].iloc[0]

    # --- 2 : Run Bucket Model & FoS ---
    init_sat = get_stable_initial_saturation(bafu_df, region_id, start_date)
    print(f"Initial Saturation (m0) from BAFU: {init_sat:.3f}")
    daily_saturation = bm.calculate_daily_saturation(
        precip_mm_day=rainfall_data,
        n=const.N,
        n_perp=const.H_PERP,
        m0=init_sat,
        drainage_rate=drainage_rate,
        et_rate=et_rate,
    )

    daily_fos = mod.compute_fos(
        m_array=daily_saturation,
        c=const.C,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )

    micp_cohesion = const.C + 15.0
    daily_fos_micp = mod.compute_fos(
        m_array=daily_saturation,
        c=micp_cohesion,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )

    # --- 4. Plotting Results ---

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Top Plot: Hyetograph (Rainfall)
    ax1.bar(time_axis, rainfall_data, color="blue", alpha=0.6)
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title("1-Year Historical Simulation (Bucket Model)")
    ax1.grid(True, alpha=0.3)

    # Middle Plot: Soil Saturation Simulated vs. Observed
    ax2.plot(
        time_axis,
        daily_saturation,
        color="purple",
        linewidth=2,
        label="Simulated Saturation",
    )

    ax2.plot(
        bafu_local["measured_at"],
        bafu_local["saturation_proxy"],
        color="black",
        linestyle="--",
        marker="o",
        markersize=3,
        label=f"BAFU Measured (Region {region_id})",
    )
    ax2.axhline(
        1.0, color="black", linestyle="--", alpha=0.3, label="Max Capacity (m=1)"
    )
    ax2.set_ylabel("Saturation Ratio (m)")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom Plot: Factor of Safety (Baseline vs MICP)
    ax3.plot(
        time_axis,
        daily_fos,
        color="red",
        linewidth=2,
        label=f"Baseline (c={const.C} kPa)",
    )
    ax3.plot(
        time_axis,
        daily_fos_micp,
        color="green",
        linewidth=2,
        label=f"MICP Treated (c={micp_cohesion} kPa)",
    )
    ax3.axhline(
        1.0,
        color="gray",
        linestyle="-.",
        linewidth=1,
        label="Failure Threshold (FoS=1.0)",
    )

    # Fill the area where failure occurs
    ax3.fill_between(
        time_axis, 0, daily_fos, where=(daily_fos <= 1.0), color="red", alpha=0.2
    )

    ax3.set_xlabel("Time (Days)")
    ax3.set_ylabel("Factor of Safety (FoS)")
    ax3.set_ylim(0.5, 4.5)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"output/simulation_{e_coord}_{n_coord}_{start_date.date()}_{end_date.date()}.png",
        dpi=300,
    )
    plt.show()


def main():
    for event in data:
        print(
            f"\nSimulating for {event['gemeinde']} ({event['start_date'].date()} to {event['end_date'].date()})"
        )
        simulate(
            e_coord=event["x_coord"],
            n_coord=event["y_coord"],
            start_date=event["start_date"],
            end_date=event["end_date"],
            region_id=event["region_id"],
        )


if __name__ == "__main__":
    main()
