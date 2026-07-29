"""
val_visuals.py - Data Visualization Utilities

Handles the generation of Matplotlib plots for historical landslide event simulations.
Keeps the main validation logic clean and modular.
"""

import matplotlib.pyplot as plt


def plot_event(rain, S, fos, bafu, date, name, idx, plot_dir, onset_val, c_val):
    """
    Generates a 3-panel subplot (Rainfall, Saturation, Factor of Safety)
    for a specific landslide event window.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Top Panel: Rainfall ---
    ax1.bar(rain.index, rain.values, color="blue", alpha=0.6)
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title(f"Event Simulation: {name} — {date.date()}")
    ax1.grid(True, alpha=0.3)

    # --- Middle Panel: Soil Saturation ---
    ax2.plot(
        S.index, S.values, color="purple", linewidth=2, label="Simulated Saturation"
    )

    if not bafu.empty:
        # Normalize BAFU nFK data
        bafu_ratio = bafu / 100.0 if bafu.max() > 2.0 else bafu
        scaled_bafu = bafu_ratio.values * onset_val
        ax2.plot(
            bafu.index,
            scaled_bafu,
            "o--",
            color="black",
            ms=4,
            label="BAFU nFK (Scaled)",
        )

    ax2.axhline(
        onset_val,
        color="orange",
        ls=":",
        label=f"Pore-Pressure Onset ({onset_val:.2f})",
    )
    ax2.axhline(
        1.0, color="gray", linestyle="--", alpha=0.3, label="Full Saturation (1.0)"
    )
    ax2.set_ylabel("Saturation Ratio")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Bottom Panel: Factor of Safety ---
    ax3.plot(
        fos.index,
        fos.values,
        color="red",
        linewidth=2,
        label=f"Baseline FoS (c'={c_val} kPa)",
    )
    ax3.axhline(
        1.0, color="gray", ls="-.", linewidth=1, label="Failure Threshold (FoS=1)"
    )
    ax3.axvline(
        date, color="black", ls="--", alpha=0.5, label="Recorded Disaster Event"
    )

    # Shade regions where FoS indicates failure
    ax3.fill_between(
        fos.index, 0, fos.values, where=(fos.values <= 1.0), color="red", alpha=0.2
    )

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Factor of Safety")
    ax3.set_ylim(0.5, 4.5)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/event_{idx:03d}_{date.date()}.png", dpi=150)
    plt.close(fig)
