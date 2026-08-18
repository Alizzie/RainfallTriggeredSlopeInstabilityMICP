import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipeline.utils import constants as auct
from core import physics
from core import constants as const
from core import data_loader as dl


def auc_score(pos, neg):
    """AUC that a positive has a LOWER FoS than a negative (landslide = low FoS)."""
    s_pos, s_neg = -np.asarray(pos), -np.asarray(
        neg
    )  # higher score = more landslide-like
    ranks = pd.Series(np.concatenate([s_pos, s_neg])).rank(method="average").values
    r_pos = ranks[: len(s_pos)].sum()
    return (r_pos - len(s_pos) * (len(s_pos) + 1) / 2) / (len(s_pos) * len(s_neg))


def roc(pos, neg):
    thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
    return ([(neg <= t).mean() for t in thr], [(pos <= t).mean() for t in thr])


def min_fos_at(x, y, date, beta_deg=const.BETA_DEG):
    """ "
    Calculates the minimum FoS within the defined time window around a given date.
    Returns NaN if insufficient historical rainfall data exists.
    """
    x, y = dl.to_lv95(x, y)
    _, drainage, et, et_amp = dl.get_region_params(x, y, auct.CALIB)

    if drainage is None:
        return np.nan

    return min_fos_for_params(x, y, date, drainage, et, et_amp, beta_deg=beta_deg)


def min_fos_for_params(
    x, y, date, drainage, et, et_amp, beta_deg=const.BETA_DEG, m0=const.M0
):
    """Lowest Factor of Safety in the window around a date, for a given calibration.

    This is the shared physics pipeline used by every temporal-validation script.
    Given one location, one date, and an explicit (drainage, et) calibration, it:
      1. loads the surrounding rainfall,
      2. simulates daily soil saturation with the bucket model,
      3. converts that to a pore-pressure ratio,
      4. computes the Factor of Safety (FoS) for each day, and
      5. returns the single lowest (most dangerous) FoS inside the +/- WINDOW_DAYS
         window. A FoS at or below 1.0 means the model predicts failure.

    Args:
        x, y: Location in LV95 coordinates.
        date: Date of interest (a real event date or a matched control date).
        drainage: Calibrated drainage rate for this location [mm/day].
        et: Calibrated evapotranspiration rate for this location [mm/day].
        beta_deg: Slope angle in degrees (default: the model's standard angle).
        m0: Initial saturation ratio for the simulation spin-up.

    Returns:
        The minimum FoS in the window, or NaN if the rainfall data is missing.
    """

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)

    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None:
        return np.nan

    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan

    # Suppress the bucket model's console output during mass-simulation runs.
    with contextlib.redirect_stdout(io.StringIO()):
        S = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=m0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
            et_amplitude=et_amp,
            day_of_year=rain.index.dayofyear.to_numpy(),
        )

    m_pp = physics.pore_pressure_ratio(S, const.S_PP_ONSET_DEFAULT)

    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)  # failure depth consistent with beta_deg

    fos = pd.Series(
        physics.compute_fos(
            m_array=m_pp,
            c=const.C,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=h_v,
            beta_rad=beta_rad,
            phi_rad=const.phi,
        ),
        index=rain.index,
    )

    win = fos.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]
    return float(win.min()) if not win.empty else np.nan


def m_pp_max_at(x, y, date, calib):
    """Max pore-pressure ratio in the +/-WINDOW around date. NaN if no data."""
    _, drainage, et, et_amp = dl.get_region_params(x, y, calib)
    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None:
        return np.nan
    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        S = physics.calculate_daily_saturation(
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
    m_pp = pd.Series(
        physics.pore_pressure_ratio(S, const.S_PP_ONSET_DEFAULT), index=rain.index
    )
    win = m_pp.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]
    return float(win.max()) if not win.empty else np.nan


def fos_from_mpp(m_pp_max, beta_deg):
    b = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(b)
    return float(
        physics.compute_fos(
            m_array=m_pp_max,
            c=const.C,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=h_v,
            beta_rad=b,
            phi_rad=const.phi,
        )
    )


def plot_roc_auc(fpr, tpr, auc, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="firebrick", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", label="random (0.50)")
    ax.set_xlabel("false positive rate (controls flagged)")
    ax.set_ylabel("true positive rate (events detected)")
    ax.set_title("ROC — temporal triggering (FoS vs matched controls)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_fos_distribution(pos, neg, auc, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 4, 41)
    ax.hist(
        neg,
        bins=bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        label="controls (no event)",
    )
    ax.hist(
        pos,
        bins=bins,
        density=True,
        alpha=0.5,
        color="firebrick",
        label="landslide events",
    )
    ax.axvline(1.0, color="black", ls="--", alpha=0.6, label="failure FoS = 1")
    ax.set_xlabel("min FoS in window")
    ax.set_ylabel("density")
    ax.set_title(f"FoS separation (AUC = {auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def classify_slope(angle_deg):
    """Categorize a sampled slope angle for the cohesion baseline"""

    if angle_deg is None or not np.isfinite(angle_deg):
        return "invalid"
    if angle_deg < auct.BETA_MIN or angle_deg > auct.BETA_MAX:
        return "invalid"
    if angle_deg > const.PHI_DEG:
        return "dry"
    return "hydro"
