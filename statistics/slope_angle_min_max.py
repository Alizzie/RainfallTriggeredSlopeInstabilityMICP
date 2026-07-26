"""
Calculate and plot FoS for each slope angle at several cohesion values
and three pore-pressure ratios.

    m = 0.0  -> dry condition
    m = 0.5  -> intermediate condition
    m = 1.0  -> saturated condition

Outputs:
    - One CSV per cohesion
    - One combined CSV
    - Three FoS-versus-slope-angle plots
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import constants as const
from core import physics

SLOPE_ANGLES = range(10, 51)

# Cohesion values in kPa
SOIL_COHESION = [0.0, 0.25, 0.5, 0.75, 1.0]

M_VALUES = {
    "dry": 0.0,
    "intermediate": 0.5,
    "saturated": 1.0,
}

OUTPUT_DIR = "output/statistics/slope_angle_min_max"


def calculate_fos(
    beta_deg: float,
    m: float,
    cohesion: float,
) -> float:
    """Calculate FoS for one angle, m value, and cohesion."""

    result = physics.compute_fos(
        m_array=np.array([m], dtype=float),
        c=cohesion,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=const.H_V,
        beta_rad=np.radians(beta_deg),
        phi_rad=const.phi,
    )

    fos = float(np.asarray(result).reshape(-1)[0])

    return fos if np.isfinite(fos) else np.nan


def cohesion_filename(cohesion: float) -> str:
    """Convert a cohesion value to a filename-safe label."""

    return f"{cohesion:.2f}".replace(".", "_")


def create_results() -> pd.DataFrame:
    """Calculate FoS values and save CSV files."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    for cohesion in SOIL_COHESION:
        cohesion_rows = []

        for beta_deg in SLOPE_ANGLES:
            row = {
                "slope_angle_deg": beta_deg,
                "cohesion_kpa": cohesion,
                "fos_m0": calculate_fos(
                    beta_deg,
                    M_VALUES["dry"],
                    cohesion,
                ),
                "fos_m0_5": calculate_fos(
                    beta_deg,
                    M_VALUES["intermediate"],
                    cohesion,
                ),
                "fos_m1": calculate_fos(
                    beta_deg,
                    M_VALUES["saturated"],
                    cohesion,
                ),
            }

            cohesion_rows.append(row)
            all_rows.append(row)

        cohesion_results = pd.DataFrame(cohesion_rows)

        cohesion_label = cohesion_filename(cohesion)

        individual_file = os.path.join(
            OUTPUT_DIR,
            f"slope_angle_fos_c_{cohesion_label}_kpa.csv",
        )

        cohesion_results.to_csv(
            individual_file,
            index=False,
            float_format="%.6f",
        )

        print(f"Saved: {os.path.abspath(individual_file)}")

    combined_results = pd.DataFrame(all_rows)

    combined_file = os.path.join(
        OUTPUT_DIR,
        "slope_angle_fos_all_cohesions.csv",
    )

    combined_results.to_csv(
        combined_file,
        index=False,
        float_format="%.6f",
    )

    print(f"Saved: {os.path.abspath(combined_file)}")

    return combined_results


def plot_fos(
    results: pd.DataFrame,
    fos_column: str,
    m: float,
    condition: str,
) -> None:
    """
    Plot FoS versus slope angle.

    Each line represents one cohesion value.
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    for cohesion in SOIL_COHESION:
        subset = results[np.isclose(results["cohesion_kpa"], cohesion)].sort_values(
            "slope_angle_deg"
        )

        ax.plot(
            subset["slope_angle_deg"],
            subset[fos_column],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"c = {cohesion:g} kPa",
        )

    # Stability threshold
    ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.5,
        label="FoS = 1",
    )

    ax.set_xlabel("Slope angle (degrees)")
    ax.set_ylabel("Factor of safety")
    ax.set_title(
        f"FoS versus slope angle\n" f"{condition.capitalize()} condition, m = {m:g}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend(title="Soil cohesion")
    fig.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"fos_vs_slope_{condition}.png",
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {os.path.abspath(output_file)}")


def main() -> None:
    results = create_results()

    plot_fos(
        results=results,
        fos_column="fos_m0",
        m=M_VALUES["dry"],
        condition="dry",
    )

    plot_fos(
        results=results,
        fos_column="fos_m0_5",
        m=M_VALUES["intermediate"],
        condition="intermediate",
    )

    plot_fos(
        results=results,
        fos_column="fos_m1",
        m=M_VALUES["saturated"],
        condition="saturated",
    )


if __name__ == "__main__":
    main()
