"""
Core physics functions for slope stability and bucket model.
"""

import numpy as np

# --- Core Functions ---


def compute_fos(c, gamma, gamma_w, h_v, beta_rad, phi_rad, m_array) -> float:
    """
    Compute the factor of safety (FOS) for a given slope at a given saturation ration m.

    FoS = [c + (y - m * y_w) * H_p * cos(beta) * tan(phi)] / [y * H_p * sin(beta)]

    Parameters:
    m (float): Saturation ratio, between 0 and 1
    c (float): Effective cohesion [kPa] (default is const.C).

    Returns:
    float: The computed factor of safety.
    """

    # Normal stress component
    sigma_n = gamma * h_v * np.cos(beta_rad) ** 2

    # Pore water pressure component
    u = m_array * gamma_w * h_v * np.cos(beta_rad) ** 2

    # Effective normal stress (not negative)
    sigma_prime = np.maximum(sigma_n - u, 0.0)

    # Shear stress (driving force)
    tau = gamma * h_v * np.sin(beta_rad) * np.cos(beta_rad)

    # fos
    fos = (c + sigma_prime * np.tan(phi_rad)) / tau

    return fos


ET_PEAK_DAY = 196  # ~mid-July: day of year when evapotranspiration peaks


def calculate_daily_saturation(
    precip_mm_day,
    n,
    n_perp,
    m0,
    s_pp_onset,
    drainage_rate=0.1,
    et_rate=2.0,
    day_of_year=None,
    et_amplitude=0.0,
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
    day_of_year (int, optional): Day of the year (1-365) for seasonal ET variation.
    et_amplitude (float): Amplitude of seasonal evapotranspiration variation, 0 - 1. With 0 (default) being no seasonal variation, and 1 being full variation.
        With a > 0: E(d) = et_rate * [1 + a * cos(2*pi*(d - ET_PEAK_DAY)/365.25)]

    Return:
        saturation ratio S = water / max_capacity, in [0, 1].
    """

    print("Running Bucket Model for Daily Saturation...")
    print(f"Initial Saturation (m0): {m0:.3f}")
    print(f"Pore-Pressure Onset: {s_pp_onset:.3f}")
    print(f"Drainage Rate: {drainage_rate:.3f} mm/day")
    print(f"Evapotranspiration Rate: {et_rate:.3f} mm/day")
    print(f"ET Amplitude: {et_amplitude:.3f}")

    # 1. Calculate the maximum capacity of the bucket in milimeters (*1000)
    max_capacity_mm = n * n_perp * 1000.0
    onset_mm = s_pp_onset * max_capacity_mm

    # 2. Convert initial saturation ratio (m0) into starting millimeters of water
    initial_water_mm = m0 * max_capacity_mm

    # 3. Initialize the moisture tracking array
    days = len(precip_mm_day)
    moisture = np.zeros(days)
    moisture[0] = initial_water_mm

    # 3b. Per-day ET
    if et_amplitude > 0.0:
        if day_of_year is None:
            raise ValueError("day_of_year must be provided when et_amplitude > 0.0")

        doy = np.asarray(day_of_year, dtype=float)

        # Calculate daily ET with seasonal variation
        et_daily = et_rate * (
            1.0 + et_amplitude * np.cos(2.0 * np.pi * (doy - ET_PEAK_DAY) / 365.25)
        )
        et_daily = np.maximum(et_daily, 0.0)
    else:
        et_daily = np.full(days, et_rate, dtype=float)

    # 4. Run daily water balance loop
    for t in range(1, days):
        excess_prev = max(0.0, moisture[t - 1] - onset_mm)  # free, drainable water
        drainage = drainage_rate * excess_prev  # drainage proportional to free water
        m = moisture[t - 1] + precip_mm_day[t] - drainage - et_daily[t]
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
