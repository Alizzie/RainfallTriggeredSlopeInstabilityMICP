"""
cohesion_sweep.py - how much cohesion does the terrain actually need?

Replaces the two saturated numbers ("+15 kPa MICP stabilises everything", "closed
forest = 15 kPa root cohesion") with a dose-response curve, and shows how sensitive
the whole spatial branch is to the cohesion assumption.

KEY IDEA - one array instead of a brute-force re-run
----------------------------------------------------
The infinite-slope FoS is LINEAR in cohesion:

    FoS(c) = (c + A) / D          A = frictional resistance [kPa]
                                  D = driving stress        [kPa]

so a pixel is unstable exactly when

    c_total < c_req := D - A      ("required cohesion")

c_req is the single number that characterises a pixel. Everything else is a
comparison against that array:

    unstable share at cohesion c      ->  mean(c < c_req)
    MICP dose-response                ->  ECDF of the deficit (c_req - c_now)
    agreement with SilvaProtect vs c  ->  same, restricted to SP pixels
    critical slope angle for a given c->  invert c_req(beta) on a beta grid

A and D are recovered FROM compute_fos itself (two evaluations per scenario), so
this script does not duplicate the physics and stays correct if core/physics.py
changes. Linearity is asserted, not assumed.

Outputs (OUTDIR):
    required_cohesion.tif          c_req per pixel, main scenario
    sweep_uniform_cohesion.csv     c -> unstable share (+ SilvaProtect metrics)
    sweep_root_scale.csv           root-cohesion scaling factor -> unstable share
    micp_dose_response.csv         added cohesion -> % of currently unstable rescued
    scenario_grid.csv              (H_PERP, m_pp) sensitivity of everything above
    fig_dose_response.png, fig_micp.png, fig_silvaprotect.png, fig_creq.png
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import physics
from core import constants as const

# --------------------------------------------------------------------------- config

SLOPE_TIF = "data/swissalti_slope/slope_deg_25m_ch.tif"
FOREST_TIF = "data/swissTLM3D/tlm_forest_25m_ch.tif"
NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"
SP_TIF = "data/hangmuren_silverprotect/hangmuren_processed.tif"  # optional
OUTDIR = "output/cohesion_sweep"

BETA_MIN, BETA_MAX = 15.0, 45.0
BARE_C = 0.25  # calibrated bare-soil cohesion [kPa]
M_PP = 1.0  # design saturation (still open problem (b))
H_MAIN = float(const.H_PERP)  # soil thickness perpendicular to slope [m]

C_GRID = np.round(np.arange(0.0, 12.0001, 0.25), 4)  # uniform total cohesion [kPa]
GAIN_GRID = np.round(np.arange(0.0, 15.0001, 0.25), 4)  # added (MICP) cohesion [kPa]
ROOT_SCALES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

ROOT_C = {  # same mapping as pipeline/root_cohesion.py
    12: 15.0,  # closed forest
    14: 10.0,  # dense small stands / copse
    13: 8.0,  # open forest
    6: 6.0,  # bush forest
}
ROOT_C_SWEEP = {
    "low": {12: 8.0, 14: 6.0, 13: 5.0, 6: 5.0},
    "mid": ROOT_C,
    "high": {12: 22.0, 14: 15.0, 13: 12.0, 6: 8.0},
}

# (H_PERP [m], m_pp [-]) combinations for the sensitivity table.
# H: 0.6 = current assumption, 0.7-1.1 = StorMe field measurements at real scars.
# m_pp: 1.0 = current worst case, lower values = a more realistic design saturation.
SCENARIO_GRID = [(h, m) for h in (0.6, 0.8, 1.0, 1.1) for m in (0.6, 0.8, 1.0)]

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------------ physics


def fos_at(beta_deg, m_pp, c, h_perp):
    """FoS for a slope-angle array at a given cohesion, saturation and soil depth."""
    beta_rad = np.radians(beta_deg)
    return physics.compute_fos(
        c=c,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        beta_rad=beta_rad,
        h_v=h_perp / np.cos(beta_rad),
        m_array=m_pp,
        phi_rad=const.phi,
    )


def linear_terms(beta_deg, m_pp, h_perp, check=True):
    """Recover driving stress D and frictional resistance A from compute_fos.

    FoS(c) = (c + A) / D, so two evaluations are enough:
        FoS(0) = A / D
        FoS(1) = (1 + A) / D   ->   D = 1 / (FoS(1) - FoS(0)),  A = FoS(0) * D
    """
    f0 = np.asarray(fos_at(beta_deg, m_pp, 0.0, h_perp), dtype=np.float64)
    f1 = np.asarray(fos_at(beta_deg, m_pp, 1.0, h_perp), dtype=np.float64)
    d = 1.0 / (f1 - f0)
    a = f0 * d
    if check:
        f5 = np.asarray(fos_at(beta_deg, m_pp, 5.0, h_perp), dtype=np.float64)
        err = float(np.nanmax(np.abs((5.0 + a) / d - f5)))
        if err > 1e-4:
            raise SystemExit(
                f"compute_fos is not linear in c (max error {err:.2e}) - "
                f"the c_req shortcut in this script is invalid."
            )
    return d.astype(np.float32), a.astype(np.float32)


def required_cohesion(beta_deg, m_pp, h_perp, check=True):
    """Cohesion [kPa] at which each pixel reaches FoS = 1. Unstable <=> c < c_req."""
    d, a = linear_terms(beta_deg, m_pp, h_perp, check=check)
    return d - a, d


# ---------------------------------------------------------------------- io helpers


def read_aligned(path, ref_shape, name):
    with rasterio.open(path) as src:
        arr = src.read(1)
    if arr.shape != ref_shape:
        raise SystemExit(f"{name} shape {arr.shape} != reference {ref_shape}")
    return arr


def cohesion_from(forest, mapping, scale=1.0):
    c = np.zeros(forest.shape, dtype=np.float32)
    for cls, val in mapping.items():
        c[forest == cls] = val * scale
    return c


def write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join("" if v is None else f"{v}" for v in row) + "\n")


def sp_metrics(ours, sp):
    """Confusion metrics of our unstable class against the SilvaProtect unstable class."""
    both = float(np.count_nonzero(ours & sp))
    only_ours = float(np.count_nonzero(ours & ~sp))
    only_sp = float(np.count_nonzero(~ours & sp))
    union = both + only_ours + only_sp
    return {
        "jaccard": both / union if union else np.nan,
        "recall_of_sp": both / (both + only_sp) if (both + only_sp) else np.nan,
        "precision_vs_sp": both / (both + only_ours) if (both + only_ours) else np.nan,
    }


# ---------------------------------------------------------------------------- main


def main():
    # ---- 1. load rasters, build the validity mask (identical to root_cohesion.py)
    with rasterio.open(SLOPE_TIF) as src:
        beta_full = src.read(1).astype(np.float32)
        profile = src.profile
        slope_nd = src.nodata

    forest_full = read_aligned(FOREST_TIF, beta_full.shape, "forest")
    nonsoil_full = read_aligned(NONSOIL_TIF, beta_full.shape, "non-soil")

    valid = np.isfinite(beta_full)
    if slope_nd is not None:
        valid &= beta_full != slope_nd
    valid &= (beta_full >= BETA_MIN) & (beta_full <= BETA_MAX)
    n_slope = int(valid.sum())
    valid &= nonsoil_full != 1
    n = int(valid.sum())

    beta = beta_full[valid]
    forest = forest_full[valid]
    del forest_full, nonsoil_full

    print(f"pixels in {BETA_MIN:.0f}-{BETA_MAX:.0f} deg : {n_slope:,}")
    print(f"soil-mantled pixels used        : {n:,}")

    # diagnostic: which land-cover codes actually occur, and how much "forest" is
    # picked up by the ROOT_C mapping. A forested share far below the ~30 % national
    # forest cover means the class codes or the rasterisation are wrong.
    codes, counts = np.unique(forest, return_counts=True)
    print("\nland-cover codes on soil pixels (code: share)")
    for code, cnt in zip(codes.tolist(), counts.tolist()):
        mark = "  <- in ROOT_C" if code in ROOT_C else ""
        print(f"  {code:>5}: {cnt / n:6.2%}{mark}")
    c_root_main = cohesion_from(forest, ROOT_C)
    print(f"forested share picked up by ROOT_C: {(c_root_main > 0).mean():.1%}")

    # ---- 2. required cohesion for the main scenario
    c_req, drive = required_cohesion(beta, M_PP, H_MAIN)
    print(
        f"\nmain scenario: H_PERP={H_MAIN:.2f} m, m_pp={M_PP:.2f}\n"
        f"  driving stress D  : {drive.min():.2f} - {drive.max():.2f} kPa\n"
        f"  required cohesion : median {np.median(c_req):.2f}, "
        f"p90 {np.percentile(c_req, 90):.2f}, max {c_req.max():.2f} kPa"
    )
    print(
        "  -> any cohesion above max(c_req) stabilises EVERY pixel; "
        "values above that are not a result, they are saturation."
    )
    del drive

    out = np.full(beta_full.shape, np.nan, dtype=np.float32)
    out[valid] = c_req
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(f"{OUTDIR}/required_cohesion.tif", "w", **p) as dst:
        dst.write(out, 1)
    del out, beta_full

    # ---- 3. optional SilvaProtect layer
    sp = None
    if os.path.exists(SP_TIF):
        with rasterio.open(SP_TIF) as src:
            sp_full = src.read(1)
        if sp_full.shape != valid.shape:
            print(f"\n[warn] SilvaProtect grid {sp_full.shape} != FoS grid - skipped")
        else:
            sp = sp_full[valid] == 1
            print(f"\nSilvaProtect unstable share: {sp.mean():.1%} of soil pixels")
        del sp_full
    else:
        print(f"\n[info] {SP_TIF} not found - SilvaProtect columns will be empty")

    # ---- 4. sweep A: uniform total cohesion -> unstable share (+ SilvaProtect)
    rows = []
    best = (None, -1.0)
    for c in C_GRID:
        ours = c < c_req
        share = float(ours.mean())
        if sp is None:
            rows.append((c, round(share, 5), None, None, None))
        else:
            m = sp_metrics(ours, sp)
            rows.append(
                (
                    c,
                    round(share, 5),
                    round(m["jaccard"], 4),
                    round(m["recall_of_sp"], 4),
                    round(m["precision_vs_sp"], 4),
                )
            )
            if m["jaccard"] > best[1]:
                best = (c, m["jaccard"])
    write_csv(
        f"{OUTDIR}/sweep_uniform_cohesion.csv",
        ["c_kPa", "unstable_share", "jaccard_sp", "recall_of_sp", "precision_vs_sp"],
        rows,
    )
    if sp is not None:
        print(
            f"\nbest agreement with SilvaProtect at a UNIFORM cohesion of "
            f"{best[0]:.2f} kPa (Jaccard {best[1]:.3f})\n"
            "  -> read as a diagnostic ('their map behaves like ours at c = x'), "
            "NOT as a calibration target."
        )

    # ---- 5. sweep B: scaling the root-cohesion assumption
    rows = []
    print("\nroot-cohesion scaling (mapping x factor, on top of bare soil)")
    header = f"{'factor':>7} {'C_closed':>9} {'unstable':>9}"
    if sp is not None:
        header += f" {'jaccard':>8} {'recall_sp':>10}"
    print(header)
    for s in ROOT_SCALES:
        c_now = BARE_C + cohesion_from(forest, ROOT_C, scale=s)
        ours = c_now < c_req
        share = float(ours.mean())
        line = f"{s:7.2f} {ROOT_C[12] * s:9.1f} {share:9.1%}"
        if sp is None:
            rows.append((s, ROOT_C[12] * s, round(share, 5), None, None))
        else:
            m = sp_metrics(ours, sp)
            rows.append(
                (
                    s,
                    ROOT_C[12] * s,
                    round(share, 5),
                    round(m["jaccard"], 4),
                    round(m["recall_of_sp"], 4),
                )
            )
            line += f" {m['jaccard']:8.3f} {m['recall_of_sp']:10.1%}"
        print(line)
    write_csv(
        f"{OUTDIR}/sweep_root_scale.csv",
        [
            "root_scale",
            "c_closed_forest_kPa",
            "unstable_share",
            "jaccard_sp",
            "recall_of_sp",
        ],
        rows,
    )

    # ---- 6. sweep C: MICP dose-response per root-cohesion variant
    print(
        "\nMICP dose-response (added cohesion needed to stabilise currently unstable pixels)"
    )
    print(
        f"{'variant':>8} {'unstable_now':>13} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6}  [kPa]"
    )
    dose = {}
    for name, mapping in [("bare", {})] + list(ROOT_C_SWEEP.items()):
        c_now = BARE_C + cohesion_from(forest, mapping)
        now = c_now < c_req
        n_now = int(now.sum())
        deficit = (c_req[now] - c_now[now]).astype(np.float32)
        curve = np.array([float((deficit <= g).mean()) for g in GAIN_GRID])
        dose[name] = (n_now, curve)
        q = np.percentile(deficit, [25, 50, 75, 90]) if n_now else [np.nan] * 4
        print(
            f"{name:>8} {n_now / n:12.1%} {q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f} {q[3]:6.2f}"
        )
        del deficit, now, c_now
    write_csv(
        f"{OUTDIR}/micp_dose_response.csv",
        ["gain_kPa"] + [f"rescued_frac_{k}" for k in dose],
        [
            [g] + [round(float(dose[k][1][i]), 5) for k in dose]
            for i, g in enumerate(GAIN_GRID)
        ],
    )

    # ---- 7. critical slope angle for a few cohesion values (main scenario)
    b_grid = np.arange(BETA_MIN, BETA_MAX + 1e-9, 0.05, dtype=np.float32)
    creq_grid, _ = required_cohesion(b_grid, M_PP, H_MAIN, check=False)
    print("\ncritical slope angle (main scenario): lowest beta that is unstable")
    for c in (0.0, BARE_C, 1.0, 2.0, 4.0, 6.0, 8.0):
        unst = creq_grid > c
        crit = float(b_grid[unst].min()) if unst.any() else np.nan
        print(
            f"  c = {c:5.2f} kPa -> beta_crit = "
            + (f"{crit:.1f} deg" if unst.any() else "none (all stable)")
        )

    # ---- 8. scenario grid: how much of this depends on H_PERP and m_pp?
    rows = []
    print("\nsensitivity of the headline numbers to H_PERP and m_pp")
    print(
        f"{'H[m]':>5} {'m_pp':>5} {'max c_req':>10} {'unst_bare':>10} {'unst_mid':>9} {'gain_50%':>9}"
    )
    c_mid = BARE_C + cohesion_from(forest, ROOT_C)
    for h, m in SCENARIO_GRID:
        creq_s, _ = required_cohesion(beta, m, h, check=False)
        unst_bare = float((BARE_C < creq_s).mean())
        now = c_mid < creq_s
        unst_mid = float(now.mean())
        gain50 = (
            float(np.percentile(creq_s[now] - c_mid[now], 50)) if now.any() else np.nan
        )
        rows.append(
            (
                h,
                m,
                round(float(creq_s.max()), 3),
                round(unst_bare, 5),
                round(unst_mid, 5),
                round(gain50, 3),
            )
        )
        print(
            f"{h:5.2f} {m:5.2f} {creq_s.max():10.2f} {unst_bare:10.1%} "
            f"{unst_mid:9.1%} {gain50:9.2f}"
        )
        del creq_s, now
    write_csv(
        f"{OUTDIR}/scenario_grid.csv",
        [
            "h_perp_m",
            "m_pp",
            "max_c_req_kPa",
            "unstable_bare",
            "unstable_mid_roots",
            "gain_median_kPa",
        ],
        rows,
    )

    # ---- 9. figures
    sweep = np.genfromtxt(
        f"{OUTDIR}/sweep_uniform_cohesion.csv", delimiter=",", names=True
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["c_kPa"], sweep["unstable_share"] * 100, color="firebrick", lw=2)
    ax.axvline(BARE_C, ls=":", color="k", lw=1)
    ax.text(BARE_C, 5, " calibrated bare soil", fontsize=8, rotation=90, va="bottom")
    ax.axvline(float(c_req.max()), ls="--", color="gray", lw=1)
    ax.text(
        float(c_req.max()),
        50,
        " saturation: nothing left to stabilise",
        fontsize=8,
        rotation=90,
        va="center",
        ha="right",
    )
    ax.set_xlabel("uniform total cohesion [kPa]")
    ax.set_ylabel("unstable pixels (FoS < 1) [%]")
    ax.set_title(f"Cohesion dose-response (H={H_MAIN:.2f} m, m_pp={M_PP:.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_dose_response.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (n_now, curve) in dose.items():
        ax.plot(
            GAIN_GRID, curve * 100, lw=2, label=f"{name} ({n_now / n:.0%} unstable now)"
        )
    ax.set_xlabel("added cohesion (MICP) [kPa]")
    ax.set_ylabel("of currently unstable pixels, rescued [%]")
    ax.set_title("MICP dose-response")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_micp.png", dpi=150)
    plt.close(fig)

    if sp is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            sweep["c_kPa"],
            sweep["jaccard_sp"],
            color="steelblue",
            lw=2,
            label="Jaccard",
        )
        ax.plot(
            sweep["c_kPa"],
            sweep["recall_of_sp"],
            color="gray",
            lw=1,
            ls="--",
            label="share of SP unstable we flag",
        )
        ax.axvline(best[0], ls=":", color="k", lw=1)
        ax.set_xlabel("uniform total cohesion [kPa]")
        ax.set_ylabel("agreement with SilvaProtect")
        ax.set_title(f"Agreement peaks at c = {best[0]:.2f} kPa (diagnostic only)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/fig_silvaprotect.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(c_req, bins=120, color="darkslategray")
    ax.set_xlabel("required cohesion c_req [kPa]")
    ax.set_ylabel("pixels")
    ax.set_title("How much cohesion the terrain needs (main scenario)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_creq.png", dpi=150)
    plt.close(fig)

    print(f"\n-> {OUTDIR}/ (4 csv, 4 png, required_cohesion.tif)")


if __name__ == "__main__":
    main()
