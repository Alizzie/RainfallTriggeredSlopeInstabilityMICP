"""
Parameters for the model.
"""

import numpy as np

# --- Geometry ---
BETA_DEG = 33.0  # slope angle [°]
H_PERP = 0.60  # perpendicular thickness [m]


# --- Derived geometry ---
beta = np.radians(BETA_DEG)  # slope angle [rad]
H_V = H_PERP / np.cos(beta)  # vertical thickness / failure depth [m]

# --- Soil properties (SM soil type) ---
C = 2  # effective cohesion [kPa]
PHI_DEG = 36.0  # effective friction angle [°]
GAMMA = 20.5  # unit weight of soil [kN/m³]
N = 0.37  # porosity [-]


# --- Derived soil properties ---
phi = np.radians(PHI_DEG)  # effective friction angle [rad]

# --- Water ---
GAMMA_W = 9.81  # unit weight of water [kN/m³]

# --- Hydraulic properties ---
# K_SAT = 1e-6  # saturated hydraulic conductivity [m/s]
K_SAT = 5e-5


# --- Initial conditions ---
# M0 = 0.60  # initial saturation ratio [-]
M0 = 0.3

# --- Pore-pressure activation threshold ---
# Effective-saturation level at which positive pore pressure is assumed to activate
# Default is a prior; the value is selected by sensitivity testing against
# Swiss saturation benchmarks (Halter et al. 2025, Landslides:
# background dry-day saturation ~0.60, low-rainfall triggering ~0.76).
S_PP_ONSET_DEFAULT = 0.70
S_PP_ONSET_SWEEP = [0.65, 0.70, 0.73, 0.76]

# Residual water content — used ONLY to convert model saturation (θ/n) into
# effective saturation Se = (θ − θr)/(θs − θr) for fair comparison with the
# benchmarks above. Small correction; state it in the writeup.
THETA_RES = 0.04
S_RES = THETA_RES / N  # ~0.11


# --- Rainfall scenarios to test [mm/h] ---
RAIN_SCENARIOS = [
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]  # covers light to intense alpine storm

# --- Time axis ---
T_HOURS = np.linspace(0, 48, 1000)  # 0 to 48 hours, 1000 points
