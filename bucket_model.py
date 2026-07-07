"""
Mass Conservation Model for Rainfall-Induced Slope Failure.

Drainage removes only the free water above the pore-pressure onset, so the
soil recedes toward that threshold rather than toward zero. Positive pore
pressure is generated only by water held ABOVE the threshold.
"""

import numpy as np


def calculate_daily_saturation(
    precip_mm_day, n, n_perp, m0, s_pp_onset, drainage_rate=0.1, et_rate=2.0
) -> np.ndarray:
    """
    Simulates daily soil moisture using a discrete bucket model (conservation of mass).

    Parameters:
    precip_mm_day (float): Daily precipitation in mm/day.
    n (float): Porosity of the soil (dimensionless).
    n_perp (float): Perpendicular thickness of the soil layer (m).
    m0 (float): Initial saturation ratio (dimensionless, between 0 and 1).
    s_pp_onset (float): Pore-pressure activation threshold (dimensionless).
    drainage_rate (float): Drainage rate in mm/day (default is 0.1 mm/day).
    et_rate (float): Evapotranspiration rate in mm/day (default is 2.0 mm/day).

    Return:
    saturation ratio S = water / max_capacity, in [0, 1].
    """

    print("Running Bucket Model for Daily Saturation...")
    print(f"Initial Saturation (m0): {m0:.3f}")
    print(f"Pore-Pressure Onset: {s_pp_onset:.3f}")
    print(f"Drainage Rate: {drainage_rate:.3f} mm/day")
    print(f"Evapotranspiration Rate: {et_rate:.3f} mm/day")

    # 1. Calculate the maximum capacity of the bucket in milimeters (*1000)
    max_capacity_mm = n * n_perp * 1000.0
    onset_mm = s_pp_onset * max_capacity_mm

    # 2. Convert initial saturation ratio (m0) into starting millimeters of water
    initial_water_mm = m0 * max_capacity_mm

    # 3. Initialize the moisture tracking array
    days = len(precip_mm_day)
    moisture = np.zeros(days)
    moisture[0] = initial_water_mm

    # 4. Run daily water balance loop
    for t in range(1, days):

        excess_prev = max(0.0, moisture[t - 1] - onset_mm)  # free, drainable water
        drainage = drainage_rate * excess_prev  # drainage proportional to free water
        m = moisture[t - 1] + precip_mm_day[t] - drainage - et_rate
        moisture[t] = max(0.0, min(m, max_capacity_mm))

    # 5. Moisture in mm convert to saturation ratio (m)
    saturation_ratio = moisture / max_capacity_mm

    return saturation_ratio


def pore_pressure_ratio(saturation, s_pp_onset) -> np.ndarray:
    """
    Pore-pressure driver for the FoS: 0 below the onset, rising to 1 at full
    saturation. Replaces raw saturation in u = m * gamma_w * H_v * cos²β.
    """
    saturation = np.asarray(saturation)
    return np.maximum(0.0, (saturation - s_pp_onset) / (1.0 - s_pp_onset))
