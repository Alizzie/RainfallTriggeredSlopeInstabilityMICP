import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const

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


def get_stable_initial_saturation(bafu_series, start_date, window=10):
    """Calculates the average saturation of the last window measurements."""
    history = bafu_series[bafu_series.index < start_date].sort_index(ascending=False)
    if len(history) < window:
        print(
            f"Warning: Not enough historical data before {start_date}. Using available data."
        )
        return history.iloc[0] if not history.empty else 0.6
    return history.head(window).mean()


def simulate(
    e_coord, n_coord, start_date: pd.Timestamp, end_date: pd.Timestamp, region_id
):
    start_date = pd.to_datetime(start_date) - pd.Timedelta(days=30)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=30)
    years = list(range(start_date.year, end_date.year + 1))

    # --- 1A: Load rainfall data (daily precipitation in mm/day) ---
    rain = dl.load_rainfall(e_coord, n_coord, years)
    if rain is None:
        print(f"No rainfall data for {e_coord}, {n_coord}")
        return

    rain = rain.loc[start_date:end_date]

    # --- 1B: Load soil moisture data (for comparison) ---
    bafu = dl.load_bafu_moisture(region_id, interpolate_daily=False)
    bafu_local = bafu.loc[start_date:end_date]

    # --- 1C: Load drainage and ET parameters for the region ---
    # filter by region_id
    drainage_rate, et_rate = dl.load_calibration_params(region_id)
    if drainage_rate is None:
        drainage_rate, et_rate = 0.5, 2.0

    # initial saturation: BAFU nFK -> bucket saturation (normalised to onset)
    init_nfk = get_stable_initial_saturation(bafu, start_date)
    init_sat = init_nfk * const.S_PP_ONSET_DEFAULT
    print(f"Initial nFK {init_nfk:.3f} -> bucket saturation {init_sat:.3f}")

    # --- 2 : Run Bucket Model & FoS ---
    daily_saturation = physics.calculate_daily_saturation(
        precip_mm_day=rain.values,
        n=const.N,
        n_perp=const.H_PERP,
        m0=init_sat,
        s_pp_onset=const.S_PP_ONSET_DEFAULT,
        drainage_rate=drainage_rate,
        et_rate=et_rate,
    )

    daily_saturation = pd.Series(daily_saturation, index=rain.index)
    m_pp = physics.pore_pressure_ratio(
        daily_saturation.values, const.S_PP_ONSET_DEFAULT
    )

    daily_fos = physics.compute_fos(
        m_array=m_pp,
        c=const.C,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )

    micp_cohesion = const.C + 15.0
    daily_fos_micp = physics.compute_fos(
        m_array=m_pp,
        c=micp_cohesion,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=const.beta,
        phi_rad=const.phi,
    )

    # --- 4. Plotting Results ---

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.bar(rain.index, rain.values, color="blue", alpha=0.6)
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title("1-Year Historical Simulation (Bucket Model)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        daily_saturation.index,
        daily_saturation.values,
        color="purple",
        linewidth=2,
        label="Simulated saturation",
    )
    if not bafu_local.empty:
        ax2.plot(
            bafu_local.index,
            bafu_local.values * const.S_PP_ONSET_DEFAULT,
            "o--",
            color="black",
            markersize=4,
            label=f"BAFU nFK (Region {region_id})",
        )

    ax2.axhline(
        const.S_PP_ONSET_DEFAULT,
        color="orange",
        linestyle=":",
        label=f"Pore-pressure onset ({const.S_PP_ONSET_DEFAULT})",
    )
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.3, label="Full saturation")
    ax2.set_ylabel("Saturation ratio")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(
        rain.index,
        daily_fos,
        color="red",
        linewidth=2,
        label=f"Baseline (c={const.C} kPa)",
    )
    ax3.plot(
        rain.index,
        daily_fos_micp,
        color="green",
        linewidth=2,
        label=f"MICP treated (c={micp_cohesion} kPa)",
    )
    ax3.axhline(1.0, color="gray", linestyle="-.", linewidth=1, label="Failure (FoS=1)")
    ax3.fill_between(
        rain.index, 0, daily_fos, where=(daily_fos <= 1.0), color="red", alpha=0.2
    )
    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Factor of Safety")
    ax3.set_ylim(0.5, 4.5)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output/diagnostics", exist_ok=True)
    plt.savefig(
        f"output/diagnostics/simulation_{e_coord}_{n_coord}_{start_date.date()}_{end_date.date()}.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    for event in data:
        print(f"\nSimulating {event['gemeinde']} ({event['start_date'].date()})")
        simulate(
            event["x_coord"],
            event["y_coord"],
            event["start_date"],
            event["end_date"],
            event["region_id"],
        )


if __name__ == "__main__":
    main()
