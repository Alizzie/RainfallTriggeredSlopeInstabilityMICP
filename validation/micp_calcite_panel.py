"""
micp_calcite_axis.py - the wiki key figure.

Panel A: MICP dose-response - added cohesion [kPa] vs share of currently unstable
         slopes rescued. One curve per H_PERP scenario.
Panel B: the same x-axis translated into a REQUIRED CALCITE CONTENT band.

Chain, both steps explicit:
  1) cohesion -> UCS   (Mohr-Coulomb, exact)
        UCS = dc * 2*cos(phi) / (1 - sin(phi))      ~= 3.92 * dc at phi = 36 deg
  2) UCS -> calcite    (literature power law, UNCERTAIN)
        UCS = k * Cc^n   ->   Cc = (UCS / k)^(1/n)
     k is derived from a literature anchor point; n ~ 2 assumed (lit. range 1.5-3.0).

CAVEAT that belongs in the caption: the anchors are calibrated at 10-15 % calcite,
we extrapolate down to ~1 %. That is below the percolation threshold where calcite
only coats grains and cohesion stays near zero, so the band likely UNDERestimates
the dose. Report it as "the mechanical demand is low", not as "we need x % calcite".

Input : output/cohesion_sweep/micp_dose_response.csv  (one file per H scenario)
Output: output/cohesion_sweep/fig_micp_calcite.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- config

# one entry per H_PERP scenario: label -> (csv path, column, colour)
SCENARIOS = {
    "H = 0.6 m": (
        "output/cohesion_sweep/micp_dose_response.csv",
        "rescued_frac_mid",
        "#c47a2c",
    ),
    "H = 1.0 m": (
        "output/cohesion_sweep_H1.0/micp_dose_response.csv",
        "rescued_frac_mid",
        "#3f6f83",
    ),
}

PHI_DEG = 36.0  # must match core/constants.py
N_EXP = 2.0  # power-law exponent, literature range 1.5 - 3.0

# literature anchors: (calcite_percent, UCS_kPa, label)
ANCHOR_WEAK = (14.98, 1030.0, "silica sand (weak response)")
ANCHOR_STRONG = (12.0, 5000.0, "gravelly sand (strong response)")

OUT = "output/cohesion_sweep/fig_micp_calcite.png"


# ------------------------------------------------------------------------ transfer


def ucs_from_cohesion(dc, phi_deg=PHI_DEG):
    """Mohr-Coulomb: unconfined compressive strength equivalent of a cohesion."""
    p = np.radians(phi_deg)
    return dc * 2.0 * np.cos(p) / (1.0 - np.sin(p))


def calcite_from_ucs(ucs, anchor, n=N_EXP):
    """Invert UCS = k*Cc^n, with k fixed by a literature anchor point."""
    cc_ref, ucs_ref, _ = anchor
    k = ucs_ref / cc_ref**n
    return np.power(np.maximum(ucs, 0.0) / k, 1.0 / n)


def main():
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )

    xmax = 0.0
    for label, (path, col, colour) in SCENARIOS.items():
        if not os.path.exists(path):
            print(f"[skip] {path} not found - run cohesion_sweep.py for this scenario")
            continue
        d = np.genfromtxt(path, delimiter=",", names=True)
        g, r = d["gain_kPa"], d[col]
        xmax = max(xmax, g.max())
        ax1.plot(g, r * 100, lw=2.2, color=colour, label=label)

        # p50 / p90 markers: first gain reaching 50 % / 90 % rescued
        for q, style in ((0.5, ":"), (0.9, "--")):
            idx = np.argmax(r >= q)
            if r[idx] >= q:
                ax1.plot(
                    [g[idx], g[idx]], [0, q * 100], style, color=colour, lw=1, alpha=0.7
                )
                ax1.annotate(
                    f"{int(q*100)} % @ {g[idx]:.1f} kPa",
                    (g[idx], q * 100),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=8,
                    color=colour,
                )

    ax1.set_ylabel("of currently unstable slopes, rescued [%]")
    ax1.set_title("MICP dose-response and its calcite equivalent")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 102)

    # ---- panel B: calcite band
    dc = np.linspace(0.01, max(xmax, 8.0), 300)
    ucs = ucs_from_cohesion(dc)
    cc_lo = calcite_from_ucs(ucs, ANCHOR_STRONG)  # strong response -> less calcite
    cc_hi = calcite_from_ucs(ucs, ANCHOR_WEAK)  # weak response  -> more calcite

    ax2.fill_between(
        dc,
        cc_lo,
        cc_hi,
        color="#8a6d3b",
        alpha=0.25,
        label="literature spread (soil-type dependent)",
    )
    ax2.plot(dc, cc_lo, color="#8a6d3b", lw=1.2)
    ax2.plot(dc, cc_hi, color="#8a6d3b", lw=1.2)
    ax2.axhspan(0, 1.0, color="#a5432f", alpha=0.10)
    ax2.text(
        dc[-1] * 0.99,
        0.5,
        "below typical percolation threshold\n(cohesion may not develop)",
        ha="right",
        va="center",
        fontsize=7.5,
        color="#a5432f",
    )

    ax2.set_xlabel("added cohesion from MICP [kPa]")
    ax2.set_ylabel("required calcite [% w/w]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_ylim(0, max(cc_hi) * 1.05)

    # secondary axis: the exact Mohr-Coulomb UCS equivalent
    sec = ax1.secondary_xaxis(
        "top", functions=(ucs_from_cohesion, lambda u: u / ucs_from_cohesion(1.0))
    )
    sec.set_xlabel(
        "UCS equivalent [kPa]  (Mohr-Coulomb, phi' = %.0f deg)" % PHI_DEG, fontsize=9
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    # ---- numbers for the caption
    print(
        f"transfer at phi' = {PHI_DEG:.0f} deg: UCS = {ucs_from_cohesion(1.0):.2f} x cohesion"
    )
    print(
        f"\n{'dc [kPa]':>9} {'UCS [kPa]':>10} {'Cc strong [%]':>14} {'Cc weak [%]':>12}"
    )
    for d_ in (1.0, 1.9, 3.9, 5.4, 9.9):
        u = ucs_from_cohesion(d_)
        print(
            f"{d_:9.2f} {u:10.1f} {calcite_from_ucs(u, ANCHOR_STRONG):14.2f} "
            f"{calcite_from_ucs(u, ANCHOR_WEAK):12.2f}"
        )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
