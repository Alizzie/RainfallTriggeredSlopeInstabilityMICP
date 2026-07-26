"""
auc_validation_uphill.py — does an UPHILL slope search fix the coordinate problem?

The fixed-33 AUC (0.82) beat the real single-pixel angle (0.61) only because the WSL
points often sit in the flat DEPOSIT, not the steep SCAR. Fix: from each point, look
UPHILL (only pixels higher than the point) within radius r and take the steepest slope
there — the scar is upslope of the deposit. Sweep r to find where this helps.

Method (elevation-gated uphill, a proxy — NOT flow-path tracing):
  candidate pixels = within radius r AND elevation >= point elevation AND slope in 15-45
  angle(r) = max slope among candidates
For each r we compute the AUC with the real angle AND, on the SAME valid subset, the
fixed-33 AUC as a fair reference. Two-edged: too large an r gives every point (event or
control) a steep uphill neighbour → separation collapses again. Watch for the PEAK.

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

_DEM = rasterio.open(auct.DEM_TIF)
_SLOPE = rasterio.open(auct.SLOPE_TIF)
PIXEL_M = abs(_SLOPE.res[0])
R_PIX = int(round(max(auct.RADII_M) / PIXEL_M))
_yy, _xx = np.mgrid[-R_PIX : R_PIX + 1, -R_PIX : R_PIX + 1]
_DIST = np.sqrt(_yy**2 + _xx**2) * PIXEL_M  # metres from centre
_ANGLE_CACHE = {}


def _read_win(src, x, y):
    try:
        row, col = src.index(x, y)
    except Exception:
        return None
    a = src.read(
        1,
        window=Window(col - R_PIX, row - R_PIX, 2 * R_PIX + 1, 2 * R_PIX + 1),
        boundless=True,
        fill_value=np.nan,
    ).astype("float32")
    if src.nodata is not None:
        a[a == src.nodata] = np.nan
    return a


def angles_by_radius(x, y):
    """dict radius_m -> uphill steepest slope (deg) in [15,45], or NaN."""
    key = (round(x), round(y))
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    dem = _read_win(_DEM, x, y)
    slp = _read_win(_SLOPE, x, y)
    out = {r: np.nan for r in auct.RADII_M}
    if dem is None or slp is None or not np.isfinite(dem[R_PIX, R_PIX]):
        _ANGLE_CACHE[key] = out
        return out
    p_elev = dem[R_PIX, R_PIX]
    uphill = dem >= p_elev
    in_band = np.isfinite(slp) & (slp >= auct.BETA_MIN) & (slp <= auct.BETA_MAX)
    for r in auct.RADII_M:
        cand = slp[(_DIST <= max(r, PIXEL_M / 2)) & uphill & in_band]
        out[r] = float(cand.max()) if cand.size else np.nan
    _ANGLE_CACHE[key] = out
    return out


def main():
    inv = dl.load_wsl_usable_inventory()
    if auct.MAX_EVENTS:
        inv = inv.sample(min(auct.MAX_EVENTS, len(inv)), random_state=0).reset_index(
            drop=True
        )
    print(
        f"{len(inv)} events; {auct.CONTROLS_PER_EVENT} controls each; radii {auct.RADII_M} m"
    )

    recs = []  # (label, m_pp_max, location_key)
    for i, ev in inv.iterrows():
        x, y = ut.to_lv95(ev["x"], ev["y"])
        key = (round(x), round(y))
        angles_by_radius(x, y)  # fills cache
        mpp = autils.m_pp_max_at(x, y, ev["date"], auct.CALIB)
        if np.isfinite(mpp):
            recs.append((1, mpp, key))
        for cd in autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT):
            g = autils.m_pp_max_at(x, y, cd, auct.CALIB)
            if np.isfinite(g):
                recs.append((0, g, key))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(inv)} processed  (records={len(recs)})")

    df = pd.DataFrame(recs, columns=["label", "mpp", "key"])

    print(
        f"\n{'radius_m':>8} {'n_valid':>8} {'AUC_real':>9} {'AUC_fix':>8} "
        f"{'medEvt':>7} {'medCtrl':>8}"
    )
    curve = []
    best = (-1, None)
    for r in auct.RADII_M:
        ang = df["key"].map(lambda k: _ANGLE_CACHE[k][r])
        m = ang.notna()
        sub = df[m]
        if len(sub) < 20:
            continue
        real = np.array(
            [autils.fos_from_mpp(mp, a) for mp, a in zip(sub["mpp"], ang[m])]
        )
        fix = np.array([autils.fos_from_mpp(mp, const.BETA_DEG) for mp in sub["mpp"]])
        lab = sub["label"].values
        auc_real = autils.auc_score(real[lab == 1], real[lab == 0])
        auc_fix = autils.auc_score(fix[lab == 1], fix[lab == 0])
        curve.append((r, len(sub), auc_real, auc_fix))
        print(
            f"{r:8d} {len(sub):8d} {auc_real:9.3f} {auc_fix:8.3f} "
            f"{np.median(real[lab==1]):7.2f} {np.median(real[lab==0]):8.2f}"
        )
        if auc_real > best[0]:
            best = (auc_real, (r, real, lab))

    curve = np.array(curve)

    # plot 1: AUC vs radius
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve[:, 0], curve[:, 2], "o-", color="firebrick", label="real angle (uphill)"
    )
    ax.plot(
        curve[:, 0],
        curve[:, 3],
        "s--",
        color="gray",
        label="fixed 33 deg (same subset)",
    )
    bi = int(np.argmax(curve[:, 2]))
    ax.scatter(
        [curve[bi, 0]],
        [curve[bi, 2]],
        s=120,
        facecolors="none",
        edgecolors="firebrick",
        linewidths=2,
        zorder=5,
        label=f"best: {int(curve[bi,0])} m (AUC {curve[bi,2]:.3f})",
    )
    ax.set_xlabel("uphill search radius (m)")
    ax.set_ylabel("AUC")
    ax.set_title("Does uphill slope search recover the geometry signal?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{auct.OUTDIR}/auc_vs_radius.png", dpi=150)
    plt.close(fig)

    # plot 2: distribution at the best radius
    r_best, real_best, lab_best = best[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 4, 41)
    ax.hist(
        real_best[lab_best == 0],
        bins=bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        label="controls",
    )
    ax.hist(
        real_best[lab_best == 1],
        bins=bins,
        density=True,
        alpha=0.5,
        color="firebrick",
        label="events",
    )
    ax.axvline(1.0, color="black", ls="--", alpha=0.6, label="FoS = 1")
    ax.set_xlabel("min FoS in window")
    ax.set_ylabel("density")
    ax.set_title(f"best uphill radius = {r_best} m  (AUC {best[0]:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{auct.OUTDIR}/fos_best_radius.png", dpi=150)
    plt.close(fig)

    print(f"\n-> {auct.OUTDIR}/auc_vs_radius.png | {auct.OUTDIR}/fos_best_radius.png")


if __name__ == "__main__":
    main()
