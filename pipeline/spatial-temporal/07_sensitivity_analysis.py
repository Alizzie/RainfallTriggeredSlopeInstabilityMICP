"""
sensitivity_analysis.py - which parameter really drives the spatial map?

Ranges are literature/field values, not made up:
    H_PERP     0.3 - 1.1 m       (StorMe measures 0.7-1.1 at real scars)
    phi        28 - 38 deg       (soil infinite-slope literature)
    gamma      17 - 22 kN/m^3    (typical soil bulk unit weight)
    c_bare     0.0 - 2.0 kPa     (calibrated value is 0.0; upper end tests
                                  "what if the calibration is wrong")
    m_pp       0.4 - 1.0         (0.4 ~ annual mean sat; 1.0 = current worst)
    root_scale 0.0 - 1.5         (roots off ... ~high ROOT_C variant)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import physics
from core import constants as const
from core import data_loader as dl
from pipeline.utils import constants as auct

# --------------------------------------------------------------------------- config

NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"
OUTDIR = "output/spatial/07_sensitivity_analysis"

BETA_MIN, BETA_MAX = auct.BETA_MIN, auct.BETA_MAX
MICP_TEST_GAIN = 2.0  # kPa - the gain used for the "MICP effect" metric

BASELINE = dict(
    H=float(const.H_PERP),  # 0.6 m
    phi_deg=float(np.degrees(const.phi)),  # 35 deg (constants.py PHI_DEG)
    gamma=float(const.GAMMA),  # ~19.5 kN/m^3
    gamma_w=float(const.GAMMA_W),  # 9.81 kN/m^3 (fixed physical constant)
    c_bare=float(const.C),  # calibrated in soil_cohesion.py (~0.0 kPa)
    m_pp=1.0,  # worst case
    root_scale=1.0,  # ROOT_C mid mapping, unscaled
)

# per-parameter sweep. baseline is inserted automatically for the OAT plots
SWEEPS = {
    "H": np.round(np.arange(0.30, 1.101, 0.05), 3),
    "phi_deg": np.round(np.arange(28.0, 38.001, 0.5), 3),
    "gamma": np.round(np.arange(17.0, 22.001, 0.25), 3),
    "c_bare": np.round(np.arange(0.0, 2.001, 0.1), 3),
    "m_pp": np.round(np.arange(0.40, 1.001, 0.05), 3),
    "root_scale": np.round(np.arange(0.0, 1.501, 0.1), 3),
}

# low / high tornado endpoints - first and last of each sweep, kept explicit so
# the plot legend prints exactly what varied. c_bare's low end (0.0) now
# equals the calibrated baseline itself -- that bar shows the effect of going
# UP from today's calibration, not a swing around some other centre.
TORNADO = {
    "H_PERP [m]": ("H", 0.3, 1.1),
    "phi [deg]": ("phi_deg", 28.0, 38.0),
    "gamma [kN/m3]": ("gamma", 17.0, 22.0),
    "c_bare [kPa]": ("c_bare", 0.0, 2.0),
    "m_pp [-]": ("m_pp", 0.4, 1.0),
    "root C scale [-]": ("root_scale", 0.0, 1.5),
}

ROOT_C = {  # matches root_cohesion.py's "mid (6-15 kPa)" tier exactly
    12: 15.0,
    14: 10.0,
    13: 8.0,
    6: 6.0,
}

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------------ helpers


def fos_at(beta_deg, c, m_pp, H, gamma, gamma_w, phi_rad):
    """FoS on the beta array under a full parameter set."""
    beta_rad = np.radians(beta_deg)
    return physics.compute_fos(
        c=c,
        gamma=gamma,
        gamma_w=gamma_w,
        beta_rad=beta_rad,
        h_v=H / np.cos(beta_rad),
        m_array=m_pp,
        phi_rad=phi_rad,
    )


def linear_terms(beta_deg, params):
    """Recover D, A so that FoS(c) = (c + A)/D at every pixel.

    Two evaluations of compute_fos are enough:
        FoS(0) = A/D
        FoS(1) = (1+A)/D    ->   D = 1/(FoS(1)-FoS(0)),  A = FoS(0)*D
    This is exact if compute_fos is linear in c, which the infinite-slope
    formula is (c only ever enters compute_fos additively -- see
    soil_cohesion.py's docstring for the same fact used directly there).
    """
    phi_rad = np.radians(params["phi_deg"])
    f0 = fos_at(
        beta_deg,
        0.0,
        params["m_pp"],
        params["H"],
        params["gamma"],
        params["gamma_w"],
        phi_rad,
    ).astype(np.float64)
    f1 = fos_at(
        beta_deg,
        1.0,
        params["m_pp"],
        params["H"],
        params["gamma"],
        params["gamma_w"],
        phi_rad,
    ).astype(np.float64)
    D = 1.0 / (f1 - f0)
    A = f0 * D
    return D.astype(np.float32), A.astype(np.float32)


def metrics(beta_deg, forest, params, gain=MICP_TEST_GAIN):
    """(baseline_unstable, micp_rescue) at this parameter set.

    baseline_unstable = share of pixels unstable with (c_bare + root_scale*ROOT_C)
    micp_rescue       = of those, share stabilised by +gain kPa extra cohesion
    """
    D, A = linear_terms(beta_deg, params)
    c_req = D - A  # cohesion needed to reach FoS = 1

    c_now = np.full(beta_deg.shape, params["c_bare"], dtype=np.float32)
    scale = params["root_scale"]
    if scale != 0.0:
        for cls, val in ROOT_C.items():
            c_now[forest == cls] += val * scale

    unstable = c_now < c_req
    n_unstable = int(unstable.sum())

    baseline_share = float(unstable.mean())
    if n_unstable == 0:
        rescue = np.nan
    else:
        c_after = c_now[unstable] + gain
        rescue = float((c_after >= c_req[unstable]).mean())
    return baseline_share, rescue


def read_aligned(path, ref_shape, name):
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != ref_shape:
        raise SystemExit(f"{name} shape {a.shape} != reference {ref_shape}")
    return a


# ---------------------------------------------------------------------------- main


def main():
    # ---- load rasters and validity mask (identical to root_cohesion.py)
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta_full = src.read(1).astype(np.float32)
        slope_nd = src.nodata

    forest_full, _, _ = dl.load_forest_raster()
    nonsoil_full = read_aligned(NONSOIL_TIF, beta_full.shape, "non-soil")

    valid = np.isfinite(beta_full)
    if slope_nd is not None:
        valid &= beta_full != slope_nd
    valid &= (beta_full >= BETA_MIN) & (beta_full <= BETA_MAX)
    valid &= nonsoil_full != 1

    beta = beta_full[valid]
    forest = forest_full[valid]
    n = int(valid.sum())
    del beta_full, forest_full, nonsoil_full
    print(f"soil pixels used: {n:,}")

    # ---- baseline
    b_share, b_rescue = metrics(beta, forest, BASELINE)
    print(
        f"\nBASELINE  H={BASELINE['H']:.2f} phi={BASELINE['phi_deg']:.1f} "
        f"gamma={BASELINE['gamma']:.1f} c={BASELINE['c_bare']:.2f} "
        f"m_pp={BASELINE['m_pp']:.2f} root_scale={BASELINE['root_scale']:.2f}"
    )
    print(
        f"  unstable share            : {b_share:6.1%}\n"
        f"  cohesion gain rescue at +{MICP_TEST_GAIN:.0f} kPa : {b_rescue:6.1%}"
    )

    # ---- OAT sweeps -> csv + line plots
    sweep_results = {}
    for name, values in SWEEPS.items():
        rows = []
        for v in values:
            p = dict(BASELINE, **{name: float(v)})
            s, r = metrics(beta, forest, p)
            rows.append((v, s, r))
        sweep_results[name] = np.array(rows)
        with open(f"{OUTDIR}/oat_{name}.csv", "w") as fh:
            fh.write(
                f"{name},unstable_share,cohesion_gain_rescue_{int(MICP_TEST_GAIN)}kPa\n"
            )
            for v, s, r in rows:
                fh.write(f"{v},{s:.6f},{'' if np.isnan(r) else f'{r:.6f}'}\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
    for ax, (name, arr) in zip(axes.flat, sweep_results.items()):
        v, s, r = arr[:, 0], arr[:, 1], arr[:, 2]
        ax.plot(v, s * 100, color="firebrick", lw=2, label="unstable %")
        ax2 = ax.twinx()
        ax2.plot(
            v,
            r * 100,
            color="steelblue",
            lw=2,
            ls="--",
            label=f"cohesion gain rescue @ +{int(MICP_TEST_GAIN)} kPa %",
        )
        ax.axvline(BASELINE[name], color="k", ls=":", lw=1)
        ax.set_xlabel(name)
        ax.set_ylabel("unstable [%]", color="firebrick")
        ax2.set_ylabel("rescued [%]", color="steelblue")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle("Parameter response (baseline marked)", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/oat_sweeps.png", dpi=150)
    plt.close(fig)

    # ---- tornado
    tornado_rows = []
    print(
        f"\n{'parameter':>17} {'low':>7} {'high':>7} "
        f"{'unst_low':>9} {'unst_hi':>9} {'d_unst':>7} "
        f"{'resc_low':>9} {'resc_hi':>9} {'d_resc':>7}"
    )
    for label, (name, low, high) in TORNADO.items():
        p_lo = dict(BASELINE, **{name: float(low)})
        p_hi = dict(BASELINE, **{name: float(high)})
        s_lo, r_lo = metrics(beta, forest, p_lo)
        s_hi, r_hi = metrics(beta, forest, p_hi)
        d_s = (s_hi - s_lo) * 100
        d_r = (r_hi - r_lo) * 100 if not (np.isnan(r_lo) or np.isnan(r_hi)) else np.nan
        tornado_rows.append((label, low, high, s_lo, s_hi, d_s, r_lo, r_hi, d_r))
        print(
            f"{label:>17} {low:7.2f} {high:7.2f} "
            f"{s_lo:9.1%} {s_hi:9.1%} {d_s:6.1f}pp "
            f"{r_lo:9.1%} {r_hi:9.1%} "
            + (f"{d_r:6.1f}pp" if not np.isnan(d_r) else "     nan")
        )

    with open(f"{OUTDIR}/tornado.csv", "w") as fh:
        fh.write(
            "parameter,low,high,unstable_low,unstable_high,delta_unstable_pp,"
            "rescue_low,rescue_high,delta_rescue_pp\n"
        )
        for row in tornado_rows:
            fh.write(
                ",".join(
                    (
                        f"{x}"
                        if isinstance(x, str)
                        else (
                            "" if isinstance(x, float) and np.isnan(x) else f"{x:.6f}"
                        )
                    )
                    for x in row
                )
                + "\n"
            )

    # sort by absolute impact on the unstable-share metric
    tornado_rows.sort(key=lambda r: abs(r[5]), reverse=True)
    labels = [r[0] for r in tornado_rows]
    d_unst = [r[5] for r in tornado_rows]
    d_resc = [r[8] for r in tornado_rows]
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.barh(y, d_unst, color="firebrick", alpha=0.8)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("delta unstable share [pp]  (high - low)")
    ax1.set_title(f"Effect on baseline\n(baseline = {b_share:.1%} unstable)")
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(y, d_resc, color="steelblue", alpha=0.8)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel(f"delta cohesion gain rescue [pp] at +{int(MICP_TEST_GAIN)} kPa")
    ax2.set_title(
        f"Effect on cohesion gain rescue\n(baseline = {b_rescue:.1%} rescued)"
    )
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Tornado - which parameter dominates each metric?", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/tornado.png", dpi=150)
    plt.close(fig)

    print(f"\n-> {OUTDIR}/  (7 csv, 2 png)")


if __name__ == "__main__":
    main()
