"""
plot_calibration_fit.py — Diagnostic: does the calibrated bucket model actually
FOLLOW the measured BAFU soil moisture?

This visualises exactly what calibrate_model.py optimised (the shape_err term):
per region it plots rainfall on top, and below the simulated saturation against
the BAFU nFK observations, with the onset (field-capacity) line for reference.

It answers the question we can't read off the results table: is a given region's
fit good, and did removing the drain penalty help or hurt the moisture match?

Reuses the loaders from calibrate_model.py and the ax2 logic from hist_simulation.py.
Run locally (needs the RhiresD + BAFU data on disk).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const

REGIONS = [46, 38]  # a "normal" region and the low-drainage outlier
YEARS = [2018, 2019]  # pick years with good BAFU coverage
OUTDIR = "output/calib_fit"
os.makedirs(OUTDIR, exist_ok=True)

for rid in REGIONS:
    drainage, et = dl.load_calibration_params(rid)
    if drainage is None:
        print(f"region {rid}: no calibration params")
        continue

    for year in YEARS:
        e, n = dl.get_region_coordinates(rid)
        rain = dl.load_rainfall(e, n, year)
        bafu = dl.load_bafu_moisture(rid, year=year, interpolate_daily=False)

        if rain is None or bafu is None or len(bafu) < 3:
            print(f"region {rid} {year}: insufficient data")
            continue

        m0 = float(bafu.iloc[0]) * const.S_PP_ONSET_DEFAULT
        sim = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=m0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
        )
        sim = pd.Series(sim, index=rain.index)

        pred_nfk = (sim / const.S_PP_ONSET_DEFAULT).clip(0, 1)
        common = bafu.index.intersection(sim.index)
        rmse = (
            float(
                np.sqrt(
                    (
                        (pred_nfk.reindex(common, method="nearest") - bafu.loc[common])
                        ** 2
                    ).mean()
                )
            )
            if len(common)
            else np.nan
        )

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        a1.bar(rain.index, rain.values, color="steelblue", alpha=0.6)
        a1.set_ylabel("rain (mm/day)")
        a1.set_title(
            f"Region {rid}, {year}  |  drainage={drainage:.2f}  et={et:.2f}  |  RMSE={rmse:.3f}"
        )

        a2.plot(
            sim.index, sim.values, color="purple", lw=2, label="simulated saturation"
        )
        a2.plot(
            bafu.index,
            bafu.values * const.S_PP_ONSET_DEFAULT,
            "o--",
            color="black",
            ms=4,
            label="BAFU nFK",
        )
        a2.axhline(const.S_PP_ONSET_DEFAULT, color="orange", ls=":", label="onset")
        a2.axhline(1.0, color="gray", ls="--", alpha=0.4, label="full saturation")
        a2.set_ylim(0, 1.15)
        a2.set_ylabel("saturation")
        a2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        out = f"{OUTDIR}/fit_region{rid}_{year}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
