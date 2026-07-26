"""
ROC/AUC analysis for the temporal factor-of-safety model using historical RhiresD rainfall data and the WSL landslide inventory.

Input: WSL landslide events and seasonally matched non-event control dates at the same locations.
Output: ROC curves, FoS-distribution plots, and AUC scores, with a target AUC greater than 0.70.

Q1. Is the minimum FoS within a defined time window around a recorded slope failure lower than the FoS during ordinary periods at the same location and in the same season?
Q2. How sensitive is the model’s discrimination performance to the assumed slope angle?

For the slope-angle sensitivity analysis, the same event and control dates are used at every angle so that the resulting ROC curves and AUC scores are directly comparable.
The model is evaluated for slope angles from 10° to 49°, and the angle producing the highest AUC is identified.
RhiresD rainfall data are loaded with a spin-up period before each evaluation window.
This allows the saturation model to account for antecedent rainfall and reduces sensitivity to the initial saturation state.
The code evalutes a WINDOW_DAYS interval around the failure date and the minimum FoS within that interval is used for the ROC/AUC analysis.

"""

import sys
import os
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from core import data_loader as dl
from core import physics
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

SLOPE_ANGLES = [i for i in range(30, 35)]
OUTPUT_DIR = f"{auct.OUTDIR}/fix_slope"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def min_fos_at(x, y, date, beta_deg=const.BETA_DEG):
    """min FoS in the +/-WINDOW around `date` at (x, y). NaN if no data."""
    x, y = ut.to_lv95(x, y)
    _, drainage, et = ut.get_region_params(x, y, auct.CALIB)

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None:
        return np.nan
    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan

    with contextlib.redirect_stdout(io.StringIO()):  # silence the bucket's prints
        S = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
        )

    m_pp = physics.pore_pressure_ratio(S, const.S_PP_ONSET_DEFAULT)
    fos = pd.Series(
        physics.compute_fos(
            m_array=m_pp,
            c=const.C,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=const.H_V,
            beta_rad=np.radians(beta_deg),
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


def prepare_cases():

    inv = dl.load_wsl_usable_inventory()

    if auct.MAX_EVENTS:
        inv = inv.sample(min(auct.MAX_EVENTS, len(inv)), random_state=0).reset_index(
            drop=True
        )

    cases = []
    for _, ev in inv.iterrows():
        controls = list(autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT))
        cases.append(
            {
                "x": ev["x"],
                "y": ev["y"],
                "date": ev["date"],
                "control_dates": controls,
            }
        )

    return cases


def run(beta_deg, cases):
    print(
        f"{len(cases)} events; " f"{auct.CONTROLS_PER_EVENT} matched controls each..."
    )

    pos, neg = [], []

    for i, case in enumerate(cases):
        f = min_fos_at(case["x"], case["y"], case["date"], beta_deg)
        if not np.isfinite(f):
            continue
        pos.append(f)

        for control_date in case["control_dates"]:
            g = min_fos_at(case["x"], case["y"], control_date, beta_deg)
            if np.isfinite(g):
                neg.append(g)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)} processed  (pos={len(pos)}, neg={len(neg)})")

    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0:
        print("not enough data to score")
        return
    auc = autils.auc_score(pos, neg)

    thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
    tpr = [(pos <= t).mean() for t in thr]  # flag as landslide when FoS <= t
    fpr = [(neg <= t).mean() for t in thr]

    print(f"beta = {beta_deg} deg")
    print(f"\npositives: {len(pos)} | controls: {len(neg)}")
    print(f"median FoS  events {np.median(pos):.2f}  |  controls {np.median(neg):.2f}")
    print(f"min FoS {np.abs(pos).min():.2f}  |  controls {np.abs(neg).min():.2f}")
    print(f"max FoS {np.abs(pos).max():.2f}  |  controls {np.abs(neg).max():.2f}")
    print(f"AUC = {auc:.3f}   (target > 0.70) \n\n")

    autils.plot_roc_auc(fpr, tpr, auc, f"{OUTPUT_DIR}/roc_auc_{beta_deg}.png")
    autils.plot_fos_distribution(
        pos,
        neg,
        auc,
        f"{OUTPUT_DIR}/fos_distributions_{beta_deg}.png",
    )


if __name__ == "__main__":
    cases = prepare_cases()

    txt_stats = open(f"{OUTPUT_DIR}/stats.txt", "a", encoding="utf-8")
    sys.stdout = txt_stats
    sys.stderr = txt_stats
    for beta in SLOPE_ANGLES:
        print(beta)
        run(beta, cases)

    txt_stats.close()
