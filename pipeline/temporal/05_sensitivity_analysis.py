"""
Tornado plot for the critical saturation threshold (S_crit): how far does
S_crit move when each parameter is varied across its literature range, one
at a time, holding everything else at the report's baseline values?
"""

import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const

GAMMA_W = 9.81

LOCKED_PARAMS_PATH = Path("output/temporal/01_calibrate_saturation/locked_params.json")
with open(LOCKED_PARAMS_PATH, "r", encoding="utf-8") as f:
    locked_params = json.load(f)
DRAINAGE_GATE = locked_params["s_onset"]
S_CRIT = locked_params["s_crit"]

BASELINE = dict(
    beta_deg=const.BETA_DEG,
    phi_deg=const.PHI_DEG,
    gamma=const.GAMMA,
    h_perp=const.H_PERP,
    n=const.N,
    onset=DRAINAGE_GATE,
    c=const.C,
)

RANGES = {
    "Slope angle (deg)": ("beta_deg", 15.0, 35.0),
    "Friction angle (deg)": ("phi_deg", 27.0, 35.0),
    "Onset": ("onset", 0.50, 0.76),
    "Cohesion c' (kPa)": ("c", 0.0, 2.96),
    "Soil unit weight (kN/m3)": ("gamma", 15.6, 19.5),
    "Soil thickness (m)": ("h_perp", 0.1, 2.0),
    "Porosity": ("n", 0.25, 0.49),
}

OUTPUT_DIR = "output/temporal/05_sensitivity_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def s_crit(beta_deg, phi_deg, gamma, h_perp, n, onset, c=0.0):
    """Critical saturation: the S at which FoS = 1.

    From FoS = A - B*m_pp = 1:
        A = c / (gamma * h_perp * sin(beta)) + tan(phi) / tan(beta)
        B = (gamma_w / gamma) * tan(phi) / tan(beta)
        m_crit = clip((A - 1) / B, 0, 1)
        S_crit = onset + m_crit * (1 - onset)
    """
    beta, phi = np.radians(beta_deg), np.radians(phi_deg)
    a = np.tan(phi) / np.tan(beta)
    if c > 0:
        a += c / (gamma * h_perp * np.sin(beta))
    b = (GAMMA_W / gamma) * (np.tan(phi) / np.tan(beta))
    m_crit = np.clip((a - 1) / b, 0.0, 1.0)
    return float(np.clip(onset + m_crit * (1 - onset), onset, 1.0))


def compute_ranges():
    baseline_value = s_crit(**BASELINE)
    rows = []
    for label, (key, lo, hi) in RANGES.items():
        params_lo = {**BASELINE, key: lo}
        params_hi = {**BASELINE, key: hi}
        s_lo, s_hi = s_crit(**params_lo), s_crit(**params_hi)
        rows.append(
            {
                "label": label,
                "lo": min(s_lo, s_hi),
                "hi": max(s_lo, s_hi),
                "range": abs(s_hi - s_lo),
            }
        )
    rows.sort(key=lambda r: r["range"], reverse=True)
    return baseline_value, rows


def plot(baseline_value, rows, outfile):
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(rows))[::-1]

    for yi, row in zip(y, rows):
        ax.barh(
            yi,
            row["hi"] - row["lo"],
            left=row["lo"],
            height=0.5,
            color="steelblue",
            edgecolor="black",
        )

    ax.axvline(
        baseline_value,
        ls="--",
        color="black",
        label=f"Baseline (beta={const.BETA_DEG}, phi={const.PHI_DEG}, onset={DRAINAGE_GATE}) = {baseline_value:.2f}",
    )
    ax.axvline(
        S_CRIT,
        ls="--",
        color="firebrick",
        label=f"Measured from WSL data = {S_CRIT:.2f}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Saturation needed to trigger failure (S_crit)")
    ax.set_title(
        "Which parameter changes the failure threshold the most?\n"
        "(each bar = full literature range, one parameter at a time)"
    )
    ax.set_xlim(0.55, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def main():
    baseline_value, rows = compute_ranges()

    with open(os.path.join(OUTPUT_DIR, "sensitivity_ranges.txt"), "w") as f:
        f.write(f"Baseline S_crit (beta=30, phi=35): {baseline_value:.4f}\n")
        for row in rows:
            f.write(
                f"  {row['label']:28s} S_crit range: {row['lo']:.3f} - {row['hi']:.3f} "
                f"(width {row['range']:.3f})\n"
            )

    plot(baseline_value, rows, os.path.join(OUTPUT_DIR, "sensitivity_tornado.png"))
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'sensitivity_tornado.png')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'sensitivity_ranges.txt')}")


if __name__ == "__main__":
    main()
