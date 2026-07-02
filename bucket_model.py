"""
Mass Conservation Model for Rainfall-Induced Slope Failure.
"""

import numpy as np


def calculate_daily_saturation(
    precip_mm_day, n, n_perp, m0, drainage_rate=0.1, et_rate=2.0
) -> np.ndarray:
    """
    Simulates daily soil moisture using a discrete bucket model (conservation of mass).

    Parameters:
    precip_mm_day (float): Daily precipitation in mm/day.
    n (float): Porosity of the soil (dimensionless).
    n_perp (float): Perpendicular thickness of the soil layer (m).
    m0 (float): Initial saturation ratio (dimensionless, between 0 and 1).
    drainage_rate (float): Drainage rate in mm/day (default is 0.1 mm/day).
    et_rate (float): Evapotranspiration rate in mm/day (default is 2.0 mm/day).

    Return:
    numpy.ndarray: Daily saturation ratio (m) ranging from 0 to 1.
    """

    print("Running Bucket Model for Daily Saturation...")
    print(f"Initial Saturation (m0): {m0:.3f}")
    print(f"Drainage Rate: {drainage_rate:.3f} mm/day")
    print(f"Evapotranspiration Rate: {et_rate:.3f} mm/day")

    # 1. Calculate the maximum capacity of the bucket in milimeters (*1000)
    max_capacity_mm = n * n_perp * 1000.0

    # 2. Convert initial saturation ratio (m0) into starting millimeters of water
    initial_water_mm = m0 * max_capacity_mm

    # 3. Initialize the moisture tracking array
    days = len(precip_mm_day)
    moisture = np.zeros(days)
    moisture[0] = initial_water_mm

    # 4. Run daily water balance loop
    for t in range(1, days):

        # Yesterday's moisture + today's rain
        m = moisture[t - 1] + precip_mm_day[t]

        # Drainage proportional to how wet the soil is
        drainage = drainage_rate * moisture[t - 1]

        # Subtract drainage and evapotranspiration
        m -= drainage + et_rate

        # Strict physical boundaries (0 to max capacity)
        m = max(0.0, min(m, max_capacity_mm))

        # Store the updated moisture
        moisture[t] = m

    # 5. Moisture in mm convert to saturation ratio (m)
    saturation_ratio = moisture / max_capacity_mm

    return saturation_ratio
