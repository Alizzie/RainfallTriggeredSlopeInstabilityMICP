"""
spatial_auc.py - V2 for the SPATIAL branch: does the map rank real WSL landslide
locations above random terrain, and what MICP gain would the real events have needed?

Score = cohesion DEFICIT:  deficit = c_req - c_now
    c_req  from output/cohesion_sweep/required_cohesion.tif  (per-pixel, main scenario)
    c_now  = BARE_C + root cohesion (same ROOT_C mapping as everywhere else)
deficit > 0  <=> pixel is unstable at the design saturation; larger = worse.
The deficit is a monotone transform of FoS, so its AUC is the map's AUC - but unlike
FoS it is directly in kPa, which lets the SAME array answer the MICP question.

Two evaluation modes, both applied identically to events and background (fairness):
    point   : deficit at the exact inventory pixel
    r<N>    : max deficit within N pixels (Chebyshev window) - addresses the documented
              scar-vs-deposit location uncertainty (Leonarduzzi 2021: 937/1354 slides
              had no unstable cell within 125 m; our own real-angle AUC dropped to
              0.61-0.69 for the same reason). At 25 m resolution, r2 = 50 m, r5 = 125 m.

Outputs (output/validation_spatial/):
    spatial_auc.csv          AUC per mode (+ n_events, n_background)
    roc_spatial.png          ROC curves for all modes
    event_dose_response.csv  gain -> share of REAL events rescued (per mode)
    fig_event_dose.png       the event-level MICP dose-response
    stats.txt                everything printed below

Honest caveats to carry into the wiki:
  - background pixels are "no slide recorded", not "no slide happened" (presence-only
    inflation, same caveat as the temporal AUC).
  - the deficit is computed at m_pp = 1.0 for every pixel; a rainfall-derived design
    saturation (open problem b) would sharpen this.
  - events dropped because they fall outside the 15-45 deg soil mask are reported,
    not silently discarded - they are the "FoS ceiling" (Leonarduzzi's ~65 %).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import utils as ut

# --------------------------------------------------------------------------- config

CREQ_TIF = "output/cohesion_sweep/required_cohesion.tif"
FOREST_TIF = "data/swissTLM3D/tlm_forest_25m_ch.tif"

OUTDIR = "output/validation_spatial"
BARE_C = 0.25

RADII_PX = (0, 2, 5)  # 0 = point, 2 = 50 m, 5 = 125 m (at 25 m pixels)
N_BACKGROUND = 50_000  # random valid pixels as the negative class
GAIN_GRID = np.round(np.arange(0.0, 8.0001, 0.25), 4)
RNG = np.random.default_rng(0)

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------------ helpers


def auc_score(pos, neg):
    """Rank-based AUC; higher score = more landslide-like."""
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank(method="average").values
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def window_max(arr, rows, cols, radius):
    """Max of arr in a (2r+1)^2 Chebyshev window around each (row, col); NaN-aware.

    radius 0 returns the point value. Points whose whole window is NaN return NaN.
    """
    if radius == 0:
        return arr[rows, cols]
    h, w = arr.shape
    out = np.full(len(rows), np.nan, dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(r - radius, 0), min(r + radius + 1, h)
        c0, c1 = max(c - radius, 0), min(c + radius + 1, w)
        win = arr[r0:r1, c0:c1]
        if np.isfinite(win).any():
            out[i] = np.nanmax(win)
    return out


# ---------------------------------------------------------------------------- main


def main():
    log = open(f"{OUTDIR}/stats.txt", "w", encoding="utf-8")

    def say(msg=""):
        print(msg)
        log.write(msg + "\n")

    # ---- 1. deficit raster = c_req - c_now
    with rasterio.open(CREQ_TIF) as src:
        c_req = src.read(1).astype(np.float32)
        transform = src.transform
    with rasterio.open(FOREST_TIF) as src:
        forest = src.read(1)
    if forest.shape != c_req.shape:
        raise SystemExit(f"forest grid {forest.shape} != c_req grid {c_req.shape}")

    c_now = np.full(c_req.shape, BARE_C, dtype=np.float32)
    deficit = c_req - c_now  # NaN outside the soil/slope mask (c_req is NaN)
    del c_req, c_now, forest

    valid = np.isfinite(deficit)
    n_valid = int(valid.sum())
    say(f"valid (soil, 15-45 deg) pixels : {n_valid:,}")
    say(f"unstable at design saturation  : {(deficit[valid] > 0).mean():.1%}")

    # ---- 2. inventory -> raster indices
    inv = dl.load_wsl_usable_inventory()
    xs, ys = zip(*(ut.to_lv95(x, y) for x, y in zip(inv["x"], inv["y"])))
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (
        (rows >= 0)
        & (rows < deficit.shape[0])
        & (cols >= 0)
        & (cols < deficit.shape[1])
    )
    say(f"\ninventory events               : {len(inv):,}")
    say(f"  outside raster extent        : {int((~inside).sum()):,}")
    rows, cols = rows[inside], cols[inside]

    on_valid = valid[rows, cols]
    say(
        f"  on a masked-out pixel (rock, <15 or >45 deg, or nodata): "
        f"{int((~on_valid).sum()):,} ({(~on_valid).mean():.1%})"
    )
    say(
        "  ^ these are the 'FoS ceiling' events - report, don't hide. "
        "(Leonarduzzi: ~65 % of slides fall in FoS-stable cells; the r>0 modes "
        "recover those whose scar is merely mislocated.)"
    )

    # ---- 3. background sample (same statistic per mode as the events)
    vr, vc = np.nonzero(valid)
    pick = RNG.choice(len(vr), size=min(N_BACKGROUND, len(vr)), replace=False)
    b_rows, b_cols = vr[pick], vc[pick]
    say(f"\nbackground pixels sampled      : {len(b_rows):,}")

    # ---- 4. AUC per mode + ROC plot
    say(f"\n{'mode':>8} {'n_events':>9} {'AUC':>7}")
    results = []
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for r_px in RADII_PX:
        pos = window_max(deficit, rows, cols, r_px)
        neg = window_max(deficit, b_rows, b_cols, r_px)
        ok_p, ok_n = np.isfinite(pos), np.isfinite(neg)
        pos, neg = pos[ok_p], neg[ok_n]
        a = auc_score(pos, neg)
        mode = "point" if r_px == 0 else f"r{r_px} ({r_px * 25} m)"
        say(f"{mode:>8} {len(pos):9,} {a:7.3f}")
        results.append((mode, len(pos), len(neg), a))

        thr = np.quantile(np.concatenate([pos, neg]), np.linspace(0, 1, 200))
        tpr = [(pos >= t).mean() for t in thr[::-1]]
        fpr = [(neg >= t).mean() for t in thr[::-1]]
        ax.plot(fpr, tpr, lw=2, label=f"{mode}: AUC {a:.3f}")

        # keep the point-mode arrays for the dose-response step
        if r_px == 0:
            pos_point = pos

    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set_xlabel("false positive rate (background flagged)")
    ax.set_ylabel("true positive rate (events flagged)")
    ax.set_title("Spatial ROC - cohesion deficit vs WSL inventory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/roc_spatial.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(results, columns=["mode", "n_events", "n_background", "auc"]).to_csv(
        f"{OUTDIR}/spatial_auc.csv", index=False
    )

    # ---- 5. event-level MICP dose-response
    # Of the events the map DOES call unstable (deficit > 0), what share would a gain
    # of g kPa have lifted to FoS >= 1 at the design saturation?
    say("\nevent-level MICP dose-response (present-day cohesion incl. roots)")
    say(f"{'mode':>8} {'n_unstable_events':>18} {'g @50%':>7} {'g @90%':>7}  [kPa]")
    curves = {}
    for r_px in RADII_PX:
        d = window_max(deficit, rows, cols, r_px)
        d = d[np.isfinite(d)]
        d_unst = d[d > 0]
        if len(d_unst) == 0:
            continue
        curve = np.array([(d_unst <= g).mean() for g in GAIN_GRID])
        mode = "point" if r_px == 0 else f"r{r_px}"
        curves[mode] = (len(d_unst), curve)
        say(
            f"{mode:>8} {len(d_unst):18,} "
            f"{np.percentile(d_unst, 50):7.2f} {np.percentile(d_unst, 90):7.2f}"
        )

    with open(f"{OUTDIR}/event_dose_response.csv", "w") as fh:
        fh.write("gain_kPa," + ",".join(f"rescued_frac_{k}" for k in curves) + "\n")
        for i, g in enumerate(GAIN_GRID):
            fh.write(
                f"{g}," + ",".join(f"{curves[k][1][i]:.5f}" for k in curves) + "\n"
            )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, (n_u, curve) in curves.items():
        ax.plot(GAIN_GRID, curve * 100, lw=2, label=f"{mode} (n={n_u:,})")
    ax.set_xlabel("added cohesion (MICP) [kPa]")
    ax.set_ylabel("of model-unstable REAL events, rescued [%]")
    ax.set_title("MICP dose-response evaluated at WSL landslide locations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_event_dose.png", dpi=150)
    plt.close(fig)

    say(f"\n-> {OUTDIR}/ (2 csv, 2 png, stats.txt)")
    log.close()


if __name__ == "__main__":
    main()
