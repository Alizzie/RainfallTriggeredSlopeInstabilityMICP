import numpy as np

import constants as const

# --- Core Functions ---


def compute_fos(m: float, c: float = const.C) -> float:
    """
    Compute the factor of safety (FOS) for a given slope at a given saturation ration m.

    FoS = [c + (y - m * y_w) * H_p * cos(beta) * tan(phi)] / [y * H_p * sin(beta)]

    Parameters:
    m (float): Saturation ratio, between 0 and 1
    c (float): Effective cohesion [kPa] (default is const.C).

    Returns:
    float: The computed factor of safety.
    """

    numerator = c + (const.GAMMA - m * const.GAMMA_W) * const.H_PERP * np.cos(
        const.beta
    ) * np.tan(const.phi)

    denominator = const.GAMMA * const.H_PERP * np.sin(const.beta)

    # Calculate the factor of safety using the formula
    fos = numerator / denominator

    return fos


def compute_m(t_h: float, m0: float = const.M0, iz_mmh: float = 20.0) -> float:
    """
    Compute saturation ratio m at time t under constant rainfall.

    m(t) = min(m0 + (Iz / K_sat) * (t / H_perp), 1.0)

    Uses H_perp directly. Water table rise is measured perpendicular to slope.

    Parameters:
    t_h (float): time since rainfall started [hours]
    m0 (float): Initial saturation ratio [-] (default is const.M0)
    iz_mmh (float): Rainfall intensity [mm/h] (default is 20.0 mm/h)

    Returns:
    m (float): Saturation ratio at time t, between 0 and 1.
    """

    iz_mps = iz_mmh / 1000.0 / 3600.0  # Convert mm/h to m/s
    t_s = t_h * 3600.0  # Convert hours to seconds
    # m = m0 + (iz_mps / const.K_SAT) * (t_s / const.H_PERP)
    m = m0 + (iz_mps * t_s) / (const.N * const.H_PERP)

    return np.clip(m, 0.0, 1.0)  # Ensure m is between 0 and 1


def find_failure_time(m0=const.M0, iz_mmh=20.0, c=const.C) -> float:
    """
    Analytically solve for the time when FoS = 1.0.

    Step 1: set FoS = 1, solve for critical saturation m*
    Step 2: invert m(t) to find t_failure

    Critical saturation m* from eq. (18) adapted to perpendicular form:
    m* = [ y * H_perp·sin(beta) - c - gamma * H_perp cos(beta)·tan(phi) ] / [ y_w·H_perp·cos(beta)·tan(phi) ]

    Parameters:
    m0 (float): Initial saturation ratio [-] (default is const.M0)
    iz_mmh (float): Rainfall intensity [mm/h] (default is 20.0 mm/h)
    c (float): Effective cohesion [kPa] (default is const

    Returns:
    t_fail_h (float): Time to failure [hours]
    """

    if iz_mmh <= 0:
        return np.inf  # no rainfall, no failure

    m_star_num = (
        const.GAMMA * const.H_PERP * np.sin(const.beta)
        - c
        - const.GAMMA * const.H_PERP * np.cos(const.beta) * np.tan(const.phi)
    )
    m_star_den = const.GAMMA_W * const.H_PERP * np.cos(const.beta) * np.tan(const.phi)
    m_star = m_star_num / m_star_den

    if m_star <= m0:
        return 0.0  # already failed at initial saturation

    if m_star >= 1.0:
        return np.inf  # never reaches critical saturation

    # Invert m(t) to get t_failure
    iz_mps = iz_mmh / 1000.0 / 3600.0
    t_fail_s = (m_star - m0) * const.K_SAT * const.H_PERP / iz_mps
    return t_fail_s / 3600.0
