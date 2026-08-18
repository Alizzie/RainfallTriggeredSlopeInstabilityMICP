"""
seasonal_cohesion.py - Phase 2 follow-up: does a FIXED cohesion's effect vary
by season?


Output (OUTPUT_DIR = output/spatial/07_seasonal_cohesion):
    seasonal_cohesion.csv    AUC/J/MCC per (season, cohesion) cell
    seasonal_cohesion.png    one line per season, per metric (3 panels)
"""

import sys
import os
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

OUTPUT_DIR = "output/spatial/08_seasonal_cohesion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COHESION_GRID_KPA = np.arange(0.0, 3.01, 0.25)

SEASON_BY_MONTH = {
    12: "Winter (DJF)",
    1: "Winter (DJF)",
    2: "Winter (DJF)",
    3: "Spring (MAM)",
    4: "Spring (MAM)",
    5: "Spring (MAM)",
    6: "Summer (JJA)",
    7: "Summer (JJA)",
    8: "Summer (JJA)",
    9: "Autumn (SON)",
    10: "Autumn (SON)",
    11: "Autumn (SON)",
}
SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
SEASON_COLORS = {
    "Winter (DJF)": "steelblue",
    "Spring (MAM)": "mediumseagreen",
    "Summer (JJA)": "firebrick",
    "Autumn (SON)": "darkorange",
}


def max_pore_pressure_ratio(x, y, date):
    """Max pore-pressure ratio in the +/- window around date, or NaN if no rainfall data.
    Same closed form as soil_cohesion.py."""
    _, drainage, et, et_amp = dl.get_region_params(x, y, auct.CALIB)
    if drainage is None:
        return np.nan

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None:
        return np.nan
    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan

    with contextlib.redirect_stdout(io.StringIO()):
        saturation = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
            et_amplitude=et_amp,
            day_of_year=rain.index.dayofyear.to_numpy(),
        )
    pore_pressure = pd.Series(
        physics.pore_pressure_ratio(saturation, const.S_PP_ONSET_DEFAULT),
        index=rain.index,
    )
    window = pore_pressure.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]
    return float(window.max()) if not window.empty else np.nan


def collect_case_control_pore_pressures_by_season():
    """{season: (event_pp_array, control_pp_array)} for the WSL case-control set.

    Each event's season comes from its own event date; its controls inherit
    that same season (see module docstring for why no separate lookup is
    needed for them).
    """
    inventory = dl.load_wsl_usable_inventory()
    by_season = {s: {"event": [], "control": []} for s in SEASON_ORDER}

    for i, event in inventory.iterrows():
        x, y = event["x"], event["y"]
        season = SEASON_BY_MONTH[event["date"].month]

        event_pp = max_pore_pressure_ratio(x, y, event["date"])
        controls = vm.find_control_date(
            x, y, event["date"], inventory, n=auct.CONTROLS_PER_EVENT
        )
        control_pps = [max_pore_pressure_ratio(x, y, c) for c in controls]

        if np.isfinite(event_pp):
            by_season[season]["event"].append(event_pp)
        by_season[season]["control"].extend(m for m in control_pps if np.isfinite(m))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(inventory)} processed")

    return {
        season: (np.array(d["event"]), np.array(d["control"]))
        for season, d in by_season.items()
    }


def fos_at_zero_cohesion_and_tau(pore_pressure_ratios, beta_deg=const.BETA_DEG):
    """(FoS at c'=0, tau) -- same exact linearity as soil_cohesion.py:
    FoS(c') = fos_at_c0 + c' / tau."""
    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)
    tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    fos_at_c0 = physics.compute_fos(
        m_array=pore_pressure_ratios,
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta_rad,
        phi_rad=const.phi,
    )
    return fos_at_c0, tau


def score_season(event_pp, control_pp):
    """AUC/J/MCC at every cohesion value, for one season's pore pressures."""
    if len(event_pp) == 0 or len(control_pp) == 0:
        return pd.DataFrame()

    fos0_events, tau = fos_at_zero_cohesion_and_tau(event_pp)
    fos0_controls, _ = fos_at_zero_cohesion_and_tau(control_pp)

    rows = []
    for cohesion in COHESION_GRID_KPA:
        fos_events = fos0_events + cohesion / tau
        fos_controls = fos0_controls + cohesion / tau
        event_flags, control_flags = fos_events <= 1.0, fos_controls <= 1.0
        rows.append(
            {
                "cohesion_kpa": cohesion,
                "auc": autils.auc_score(fos_events, fos_controls),
                "youden_j": vm.youden_j(event_flags, control_flags),
                "mcc": vm.mcc(event_flags, control_flags),
                "n_events": len(event_pp),
                "n_controls": len(control_pp),
            }
        )
    return pd.DataFrame(rows)


def main():
    print(
        f"Fixed slope angle: {const.BETA_DEG}\u00b0 (same as soil_cohesion.py's winner)"
    )
    print("Collecting pore pressures per event/control, bucketed by season...")
    by_season = collect_case_control_pore_pressures_by_season()

    for season in SEASON_ORDER:
        n_e, n_c = len(by_season[season][0]), len(by_season[season][1])
        print(f"  {season:>15}: {n_e} events, {n_c} controls")

    all_rows = []
    for season in SEASON_ORDER:
        event_pp, control_pp = by_season[season]
        curve = score_season(event_pp, control_pp)
        if curve.empty:
            print(f"[warn] {season}: no usable data, skipped")
            continue
        curve = curve.assign(season=season)
        all_rows.append(curve)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(f"{OUTPUT_DIR}/seasonal_cohesion.csv", index=False)
    print("\n--- Best Youden's J per season ---")
    for season in SEASON_ORDER:
        sub = result[result.season == season]
        if sub.empty:
            continue
        best = sub.loc[sub["youden_j"].idxmax()]
        print(
            f"{season:>15}: best J={best.youden_j:.3f} at c'={best.cohesion_kpa:.2f} kPa "
            f"(AUC at c'=0: {sub.loc[sub.cohesion_kpa == 0, 'auc'].values[0]:.3f})"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, (col_name, title) in zip(
        axes, [("auc", "AUC"), ("youden_j", "Youden's J"), ("mcc", "MCC")]
    ):
        for season in SEASON_ORDER:
            sub = result[result.season == season].sort_values("cohesion_kpa")
            if sub.empty:
                continue
            ax.plot(
                sub.cohesion_kpa,
                sub[col_name],
                "o-",
                color=SEASON_COLORS[season],
                label=f"{season} (n={int(sub.n_events.iloc[0])})",
            )
        ax.set_xlabel("cohesion c' (kPa)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Fixed cohesion's effect by season (beta={const.BETA_DEG}\u00b0 fixed, "
        f"cohesion fixed)",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/seasonal_cohesion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {OUTPUT_DIR}/ (seasonal_cohesion.csv, seasonal_cohesion.png)")


if __name__ == "__main__":
    main()
