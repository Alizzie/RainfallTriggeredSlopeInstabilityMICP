import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import constants as const
import bondo_case_study.saturation as sat
import model

# --- Initial Setup ---
time_hours = const.T_HOURS
initial_rain = 30.0
initial_m0 = const.M0

# --- Figure and Axis Setup ---
fig, ax = plt.subplots(figsize=(10, 8))
# We increase the bottom margin to 0.45 to make room for 5 sliders
plt.subplots_adjust(left=0.1, bottom=0.45)

# Initial data generation
dynamic_m = sat.get_dynamic_saturation(
    initial_m0, initial_rain, time_hours, const.N, const.H_PERP
)
static_m = sat.get_static_saturation(initial_m0, time_hours)

dynamic_fos = model.compute_fos(
    const.C, const.GAMMA, const.GAMMA_W, const.H_V, const.beta, const.phi, dynamic_m
)
static_fos = model.compute_fos(
    const.C, const.GAMMA, const.GAMMA_W, const.H_V, const.beta, const.phi, static_m
)

# Plotting
(line_dyn,) = ax.plot(
    time_hours,
    dynamic_fos,
    label="Dynamic Rainfall Triggered",
    color="blue",
    linewidth=2,
)
(line_stat,) = ax.plot(
    time_hours,
    static_fos,
    label="Static Baseline (No Rain)",
    color="grey",
    linestyle="--",
)
ax.axhline(1.0, color="red", linestyle=":", label="FoS = 1.0 (Failure)")

ax.set_title("Interactive Infinite Slope Stability")
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("Factor of Safety (FoS)")
ax.set_ylim(0.5, 2.5)
ax.legend()
ax.grid(True)

# --- Interactive Widgets (Axes setup) ---
# [left, bottom, width, height]
ax_rain = plt.axes([0.15, 0.30, 0.65, 0.03])
ax_m0 = plt.axes([0.15, 0.25, 0.65, 0.03])
ax_beta = plt.axes([0.15, 0.15, 0.65, 0.03])  # Slope angle
ax_phi = plt.axes([0.15, 0.10, 0.65, 0.03])  # Friction angle
ax_c = plt.axes([0.15, 0.05, 0.65, 0.03])  # Cohesion

# Create sliders
slider_rain = Slider(ax_rain, "Rain (mm/h)", 0.1, 100.0, valinit=initial_rain)
slider_m0 = Slider(ax_m0, "Init Sat (m0)", 0.0, 1.0, valinit=initial_m0)
slider_beta = Slider(ax_beta, "Slope Angle (°)", 20.0, 50.0, valinit=const.BETA_DEG)
slider_phi = Slider(ax_phi, "Friction (°)", 20.0, 45.0, valinit=const.PHI_DEG)
slider_c = Slider(ax_c, "Cohesion (kPa)", 0.0, 15.0, valinit=const.C)


# --- Update Function ---
def update(val):
    # Get current values from all sliders
    current_rain = slider_rain.val
    current_m0 = slider_m0.val
    current_c = slider_c.val

    # Convert angles to radians
    current_beta_rad = np.radians(slider_beta.val)
    current_phi_rad = np.radians(slider_phi.val)

    # CRITICAL: Recalculate vertical thickness (H_v) because beta changed!
    current_h_v = const.H_PERP / np.cos(current_beta_rad)

    # Recalculate saturation arrays
    new_dyn_m = sat.get_dynamic_saturation(
        current_m0, current_rain, time_hours, const.N, const.H_PERP
    )
    new_stat_m = sat.get_static_saturation(current_m0, time_hours)

    # Recalculate FoS arrays with new geometry
    new_dyn_fos = model.compute_fos(
        current_c,
        const.GAMMA,
        const.GAMMA_W,
        current_h_v,
        current_beta_rad,
        current_phi_rad,
        new_dyn_m,
    )
    new_stat_fos = model.compute_fos(
        current_c,
        const.GAMMA,
        const.GAMMA_W,
        current_h_v,
        current_beta_rad,
        current_phi_rad,
        new_stat_m,
    )

    # Update plot lines
    line_dyn.set_ydata(new_dyn_fos)
    line_stat.set_ydata(new_stat_fos)

    fig.canvas.draw_idle()


# Attach the update function to all sliders
slider_rain.on_changed(update)
slider_m0.on_changed(update)
slider_beta.on_changed(update)
slider_phi.on_changed(update)
slider_c.on_changed(update)

plt.show()
