import numpy as np

import constants as const

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
