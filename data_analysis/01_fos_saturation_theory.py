"""
01_fos_saturation_theory.py - Cohesion / slope / saturation decision engine.

Data-free companion to the temporal-validation scripts. Everything here follows
straight from the infinite-slope physics in core.physics, evaluated over a grid
of slope angle (beta) and effective cohesion (c'). It answers the three questions
that the cap-vs-cohesion choice hinges on:

  1. For a given (beta, c'), at what RAW saturation S does the slope fail (FoS=1)?
     -> the "critical saturation surface", the physical bridge to rainfall.
  2. Which (beta, c') combinations fail while dry (unphysical for a real slope
     that is standing today) vs. never fail even fully saturated (no trigger)?
  3. How much cohesion must MICP supply to move a slope of a given angle from
     one regime to another (stand dry / survive full saturation)?

Sign conventions and the perpendicular->vertical depth handling (h_v = H_perp/cos b)
are taken verbatim from core.physics.compute_fos, so the outputs are consistent
with the validation pipeline rather than a parallel re-derivation.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import constants as const
from core import physics

OUTPUT_DIR = "output/data_analysis/01_fos_saturation_theory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Observed / benchmark reference lines (from your earlier notes + Schaller inventories)
S_BACKGROUND = 0.60  # typical dry-day saturation
S_FAILURE_BM = 0.76  # low-rainfall triggering benchmark
ONSET = const.S_PP_ONSET_DEFAULT

# Grids
BETA_GRID = np.linspace(const.BETA_MIN, const.BETA_MAX + 10, 121)  # 15..45 deg
C_GRID = np.arange(0.0, 12.01, 0.25)  # kPa (matches val_constants)


# ---------------------------------------------------------------------------
# Closed-form helpers (FoS is linear in the pore-pressure ratio while the
# effective normal stress stays positive, which it does for every case here).
# ---------------------------------------------------------------------------
def _stresses(beta_rad):
    """Dry normal stress N0, driving shear tau, and pore-pressure lever, all [kPa]."""
    h_v = const.H_PERP / np.cos(beta_rad)
    N0 = const.GAMMA * h_v * np.cos(beta_rad) ** 2  # = gamma * H_perp * cos b
    tau = (
        const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    )  # = gamma * H_perp * sin b
    lever = const.GAMMA_W * h_v * np.cos(beta_rad) ** 2  # multiplies m_pp in u
    return N0, tau, lever


def dry_fos(beta_rad, c):
    """FoS with no pore pressure (m_pp = 0)."""
    N0, tau, _ = _stresses(beta_rad)
    return (c + N0 * np.tan(const.phi)) / tau


def m_pp_crit(beta_rad, c):
    """Pore-pressure ratio m_pp in [0,1] at which FoS = 1 (may fall outside [0,1])."""
    N0, tau, lever = _stresses(beta_rad)
    return (N0 - (tau - c) / np.tan(const.phi)) / lever


def s_crit(beta_rad, c, onset=ONSET):
    """Raw saturation at failure, inverting pore_pressure_ratio. NaN outside [0,1]."""
    mpp = m_pp_crit(beta_rad, c)
    s = onset + (1.0 - onset) * mpp
    return np.where((mpp >= 0.0) & (mpp <= 1.0), s, np.nan)


def c_for_dry_fos(beta_rad, target):
    """Cohesion needed so the DRY FoS hits `target` (0 if friction already suffices)."""
    N0, tau, _ = _stresses(beta_rad)
    return np.maximum(target * tau - N0 * np.tan(const.phi), 0.0)


def c_for_saturated_stability(beta_rad, target=1.0):
    """Cohesion needed so FoS(m_pp = 1) >= target (survives full pore pressure)."""
    N0, tau, lever = _stresses(beta_rad)
    sigma_sat = np.maximum(N0 - lever, 0.0)
    return np.maximum(target * tau - sigma_sat * np.tan(const.phi), 0.0)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary():
    repose = const.PHI_DEG
    print("=" * 70)
    print(
        f"phi' = {const.PHI_DEG} deg  ->  dry cohesionless repose angle = {repose} deg"
    )
    print(f"H_perp = {const.H_PERP} m,  gamma = {const.GAMMA} kN/m3,  onset = {ONSET}")
    print("=" * 70)

    # cross-check the closed form against core.physics.compute_fos at one point
    b = np.radians(const.BETA_DEG)
    h_v = const.H_PERP / np.cos(b)
    check = physics.compute_fos(0.0, const.GAMMA, const.GAMMA_W, h_v, b, const.phi, 0.0)
    print(
        f"[check] dry FoS at beta={const.BETA_DEG}, c=0: "
        f"closed-form {dry_fos(b, 0.0):.4f} vs compute_fos {check:.4f}"
    )
    print("-" * 70)

    print(f"{'beta':>5} {'dryFoS(c=0)':>11} {'m_pp_crit':>10} {'S_crit':>8}  regime")
    for bd in [30, 33, 35, 36, 38, 40]:
        b = np.radians(bd)
        d = dry_fos(b, 0.0)
        mpp = m_pp_crit(b, 0.0)
        if mpp < 0:
            regime, sc = "FAILS DRY (needs c')", float("nan")
        elif mpp > 1:
            regime, sc = "never fails (even saturated)", float("nan")
        else:
            regime = "rain-triggered"
            sc = ONSET + (1 - ONSET) * mpp
        sc_s = f"{sc:8.3f}" if np.isfinite(sc) else f"{'--':>8}"
        print(f"{bd:5d} {d:11.3f} {mpp:10.3f} {sc_s}  {regime}")

    print("-" * 70)
    print("Cohesion targets [kPa] (exact, this model's depth convention):")
    print(
        f"{'beta':>5} {'c: dry FoS=1':>13} {'c: dry FoS=1.3':>15} {'c: sat-stable':>14}"
    )
    for bd in [33, 36, 40]:
        b = np.radians(bd)
        print(
            f"{bd:5d} {c_for_dry_fos(b, 1.0):13.2f} "
            f"{c_for_dry_fos(b, 1.3):15.2f} {c_for_saturated_stability(b):14.2f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Figure 1: FoS vs raw saturation, two cohesion panels
# ---------------------------------------------------------------------------
def fig_fos_vs_saturation():
    S = np.linspace(0.0, 1.0, 400)
    betas = [30, 33, 36, 40]
    c_levels = [0.0, 2.0]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(betas)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, c in zip(axes, c_levels):
        for bd, col in zip(betas, colors):
            b = np.radians(bd)
            h_v = const.H_PERP / np.cos(b)
            m_pp = physics.pore_pressure_ratio(S, ONSET)
            fos = physics.compute_fos(
                c, const.GAMMA, const.GAMMA_W, h_v, b, const.phi, m_pp
            )
            ax.plot(S, fos, color=col, lw=2, label=f"{bd} deg")
        ax.axhline(1.0, color="black", ls="-.", lw=1)
        ax.axvline(ONSET, color="orange", ls=":", lw=1.2, label=f"onset {ONSET}")
        ax.axvspan(
            S_BACKGROUND,
            S_FAILURE_BM,
            color="grey",
            alpha=0.12,
            label="bg->trigger band",
        )
        ax.set_xlabel("raw saturation  S")
        ax.set_title(f"c' = {c:.0f} kPa")
        ax.set_ylim(0, 3)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Factor of Safety")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "FoS vs saturation — flat until the pore-pressure onset, then declining",
        fontweight="bold",
    )
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/fig1_fos_vs_saturation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: critical-saturation surface over (beta, c')
# ---------------------------------------------------------------------------
def fig_critical_surface():
    B, C = np.meshgrid(BETA_GRID, C_GRID)
    Brad = np.radians(B)
    MPP = m_pp_crit(Brad, C)
    S = ONSET + (1 - ONSET) * MPP

    dry_fail = MPP < 0.0  # fails before any pore pressure
    never = MPP > 1.0  # stable even fully saturated
    trig = (~dry_fail) & (~never)

    Splot = np.where(trig, S, np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    pcm = ax.pcolormesh(
        B, C, Splot, cmap="viridis", shading="auto", vmin=ONSET, vmax=1.0
    )
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("raw saturation at failure  S_crit")

    # regime overlays
    ax.contourf(
        B, C, dry_fail.astype(float), levels=[0.5, 1.5], colors="none", hatches=["xxx"]
    )
    ax.contourf(B, C, never.astype(float), levels=[0.5, 1.5], colors=["#d9d9d9"])

    # useful contours in raw saturation
    cs = ax.contour(
        B, C, Splot, levels=[0.65, 0.70, 0.76, 0.85], colors="white", linewidths=1.0
    )
    ax.clabel(cs, fmt="%.2f", fontsize=8)

    # release-slope band + operating point
    ax.axvline(const.PHI_DEG, color="red", ls="--", lw=1)
    ax.text(
        const.PHI_DEG + 0.2,
        C_GRID.max() * 0.94,
        "effective friction angle (phi)'",
        color="red",
        fontsize=9,
        rotation=90,
        va="top",
    )
    ax.scatter(
        [const.BETA_DEG],
        [const.C],
        s=50,
        color="red",
        zorder=5,
        edgecolor="white",
        label=f"current ({const.BETA_DEG} deg, c'={const.C})",
    )

    ax.set_xlabel("slope angle  beta  [deg]")
    ax.set_ylabel("effective cohesion  c'  [kPa]")
    ax.set_title(
        f"Critical saturation surface ({const.H_PERP} m)\n"
        "hatched = fails while dry   |   grey = stable even fully saturated"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_critical_saturation_surface_{const.H_PERP}m.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: cohesion the treatment must supply, per slope angle
# ---------------------------------------------------------------------------
def fig_cohesion_targets():
    b = np.radians(BETA_GRID)
    c_dry1 = c_for_dry_fos(b, 1.0)
    c_dry13 = c_for_dry_fos(b, 1.3)
    c_sat = c_for_saturated_stability(b, 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(BETA_GRID, c_dry1, color="firebrick", lw=2, label="stand dry (FoS=1)")
    ax.plot(
        BETA_GRID,
        c_dry13,
        color="darkorange",
        lw=2,
        ls="--",
        label="dry margin (FoS=1.3)",
    )
    ax.plot(BETA_GRID, c_sat, color="steelblue", lw=2, label="survive full saturation")
    ax.fill_between(
        BETA_GRID,
        c_dry1,
        c_sat,
        where=(c_sat > c_dry1),
        color="steelblue",
        alpha=0.10,
        label="rain-triggered window",
    )

    ax.axvline(const.PHI_DEG, color="grey", ls=":", lw=1)
    for bd in (33, 36):
        ax.axvline(bd, color="black", ls=":", lw=0.8, alpha=0.5)
        ax.annotate(
            f"{bd} deg", (bd, ax.get_ylim()[1]), fontsize=8, ha="center", va="bottom"
        )
    ax.set_xlabel("slope angle  beta  [deg]")
    ax.set_ylabel("required effective cohesion  c'  [kPa]")
    ax.set_ylim(0, C_GRID.max())
    ax.set_title("Cohesion the soil (or MICP) must supply, by slope angle")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_cohesion_targets.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    print_summary()
    p1 = fig_fos_vs_saturation()
    p2 = fig_critical_surface()
    p3 = fig_cohesion_targets()
    print("\nsaved:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
