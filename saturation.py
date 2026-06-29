# saturation.py
import numpy as np


def get_static_saturation(m_value, time_array):
    """
    Returns an array of constant saturation over time.
    """
    return np.full_like(time_array, m_value)


def get_dynamic_saturation(m0, i_z_mm_hr, time_hours, n, h_perp):
    """
    Calculates time-dependent saturation using a volumetric water balance.
    i_z_mm_hr: rainfall intensity in mm/h
    time_hours: numpy array of time in hours
    """
    # Convert intensity from mm/h to m/h
    i_z_m_hr = i_z_mm_hr / 1000.0

    # Calculate added water volume per unit area over time
    added_water = i_z_m_hr * time_hours

    # Available void space per square meter
    void_space = n * h_perp

    # Calculate saturation ratio
    m = m0 + (added_water / void_space)

    # Cap saturation at 1.0 (fully saturated)
    return np.clip(m, 0.0, 1.0)
