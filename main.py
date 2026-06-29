import numpy as np
import matplotlib.pyplot as plt
import constants as const
import model as mod

# =============================================================================
# SECTION 1 — PARAMETER SUMMARY + SANITY CHECKS
# =============================================================================

print("=" * 60)
print("SLOPE INSTABILITY MODEL — PARAMETER SUMMARY")
print("=" * 60)
print(f"  β        : {const.BETA_DEG}°  ({const.beta:.4f} rad)")
print(f"  H_⊥      : {const.H_PERP} m")
print(f"  c'       : {const.C} kPa")
print(f"  φ'       : {const.PHI_DEG}°  ({const.phi:.4f} rad)")
print(f"  γ        : {const.GAMMA} kN/m³")
print(f"  γ_w      : {const.GAMMA_W} kN/m³")
print(f"  K_sat    : {const.K_SAT:.1e} m/s")
print(f"  m₀       : {const.M0}")

# --- Dry stability check ---
# With c'=0, FoS_dry = tan(φ')/tan(β). Must be > 1.0 for slope to exist.
dry_check = np.tan(const.phi) / np.tan(const.beta)
print(f"\n  Dry stability check tan(φ')/tan(β) : {dry_check:.4f}", end="")
print("  ✓ stable" if dry_check > 1.0 else "  ✗ FAILS EVEN DRY — check parameters")

# --- Critical saturation m* ---
m_star_num = (
    const.GAMMA * const.H_PERP * np.sin(const.beta)
    - const.C
    - const.GAMMA * const.H_PERP * np.cos(const.beta) * np.tan(const.phi)
)
m_star_den = const.GAMMA_W * const.H_PERP * np.cos(const.beta) * np.tan(const.phi)
m_star = m_star_num / m_star_den

print(f"\n  Critical saturation m*  : {m_star:.4f}")
if m_star < 0:
    print("  NOTE: m* < 0 → slope never fails under any rainfall (too strong)")
elif m_star > 1:
    print("  NOTE: m* > 1 → slope never fails even fully saturated")
else:
    print(f"  Slope fails when m rises from m₀={const.M0} to m*={m_star:.4f}")

# --- FoS at key saturation levels ---
print(f"\n  FoS values:")
for m_val in [0.00, 0.30, 0.60, 0.76, 1.00]:
    f = mod.compute_fos(m_val)
    flag = "  ← FAILS" if f < 1.0 else ""
    print(f"    m={m_val:.2f} → FoS={f:.4f}{flag}")

# --- Saturation timescales ---
print(f"\n  Time to full saturation from m₀={const.M0} at K_sat={const.K_SAT:.1e} m/s:")
for Iz in const.RAIN_SCENARIOS:
    if Iz == 0:
        continue
    Iz_mps = Iz / 1000.0 / 3600.0
    t_sat_h = (1.0 - const.M0) * const.K_SAT * const.H_PERP / Iz_mps / 3600.0
    print(f"    {Iz:>5} mm/h → {t_sat_h:.2f} h")

# --- Failure times ---
print(f"\n  Failure times (m₀={const.M0}):")
for Iz in const.RAIN_SCENARIOS:
    tf = mod.find_failure_time(m0=const.M0, iz_mmh=Iz)
    if tf == np.inf:
        tag = "no failure"
    elif tf == 0.0:
        tag = "already failed at t=0"
    else:
        tag = f"{tf:.2f} h"
    print(f"    {Iz:>5} mm/h → {tag}")

# =============================================================================
# SECTION 2 — FIGURE 1: FoS vs time, varying rainfall (m0 fixed)
# =============================================================================

colors = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"]

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle(
    f"Fig 1 — Effect of rainfall intensity\n"
    f"β={const.BETA_DEG}°  H⊥={const.H_PERP}m  φ'={const.PHI_DEG}°  "
    f"c'={const.C}kPa  γ={const.GAMMA}kN/m³  K_sat={const.K_SAT:.0e}m/s  m₀={const.M0}",
    fontsize=10,
)

ax1 = axes1[0]
ax1.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7, label="FoS=1.0 (failure)")
ax1.axhline(1.5, color="gray", lw=0.8, ls=":", alpha=0.5, label="FoS=1.5 (min. safe)")
ax1.fill_between([0, 48], 0, 1.0, color="#E24B4A", alpha=0.06)

for Iz, col in zip(const.RAIN_SCENARIOS, colors):
    m_t = mod.compute_m(const.T_HOURS, m0=const.M0, iz_mmh=Iz)
    fos_t = mod.compute_fos(m=m_t)
    ax1.plot(const.T_HOURS, fos_t, color=col, lw=2.0, label=f"{Iz} mm/h")
    tf = mod.find_failure_time(m0=const.M0, iz_mmh=Iz)
    if 0 < tf <= const.T_HOURS[-1]:
        ax1.axvline(tf, color=col, lw=0.8, ls=":", alpha=0.5)
        ax1.scatter([tf], [1.0], color=col, s=60, zorder=5)

ax1.set_xlabel("Time since rainfall onset (hours)", fontsize=11)
ax1.set_ylabel("Factor of Safety (FoS)", fontsize=11)
ax1.set_title("FoS over time — varying rainfall intensity", fontsize=11)
ax1.set_xlim(0, 48)
ax1.set_ylim(0, 2.5)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2, lw=0.5)

ax2 = axes1[1]
for Iz, col in zip(const.RAIN_SCENARIOS, colors):
    m_t = mod.compute_m(const.T_HOURS, m0=const.M0, iz_mmh=Iz)
    ax2.plot(const.T_HOURS, m_t, color=col, lw=2.0, label=f"{Iz} mm/h")

ax2.axhline(
    const.M0,
    color="gray",
    lw=0.8,
    ls=":",
    alpha=0.6,
    label=f"m₀={const.M0} (antecedent, Halter 2025 [6])",
)
ax2.axhline(
    1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="m=1.0 (full saturation)"
)
if 0 < m_star < 1:
    ax2.axhline(
        m_star,
        color="#A32D2D",
        lw=1.0,
        ls="-.",
        alpha=0.7,
        label=f"m*={m_star:.2f} (critical — failure)",
    )

ax2.set_xlabel("Time since rainfall onset (hours)", fontsize=11)
ax2.set_ylabel("Saturation ratio m(t)", fontsize=11)
ax2.set_title("Saturation ratio over time", fontsize=11)
ax2.set_xlim(0, 48)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, lw=0.5)

plt.tight_layout()
plt.savefig("fig1_rainfall_intensity.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# SECTION 3 — FIGURE 2: FoS vs time, varying antecedent saturation (Iz fixed)
# =============================================================================

m0_scenarios = {
    "Dry (m₀=0.30)": 0.30,
    "Background (m₀=0.60)": 0.60,
    "Pre-failure (m₀=0.76)": 0.76,
}
colors_m0 = ["#378ADD", "#EF9F27", "#E24B4A"]
Iz_fixed = 20  # mm/h

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(
    f"Fig 2 — Effect of antecedent saturation (Halter 2025 [6])\n"
    f"β={const.BETA_DEG}°  H⊥={const.H_PERP}m  φ'={const.PHI_DEG}°  "
    f"c'={const.C}kPa  γ={const.GAMMA}kN/m³  K_sat={const.K_SAT:.0e}m/s  "
    f"Rainfall={Iz_fixed} mm/h",
    fontsize=10,
)

ax3 = axes2[0]
ax3.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7, label="FoS=1.0 (failure)")
ax3.axhline(1.5, color="gray", lw=0.8, ls=":", alpha=0.5, label="FoS=1.5 (min. safe)")
ax3.fill_between([0, 48], 0, 1.0, color="#E24B4A", alpha=0.06)

for (label, m0_val), col in zip(m0_scenarios.items(), colors_m0):
    m_t = mod.compute_m(const.T_HOURS, m0=m0_val, iz_mmh=Iz_fixed)
    fos_t = mod.compute_fos(m=m_t)
    ax3.plot(const.T_HOURS, fos_t, color=col, lw=2.0, label=label)
    tf = mod.find_failure_time(m0=m0_val, iz_mmh=Iz_fixed)
    if 0 < tf <= const.T_HOURS[-1]:
        ax3.axvline(tf, color=col, lw=0.8, ls=":", alpha=0.5)
        ax3.scatter([tf], [1.0], color=col, s=60, zorder=5)

ax3.set_xlabel("Time since rainfall onset (hours)", fontsize=11)
ax3.set_ylabel("Factor of Safety (FoS)", fontsize=11)
ax3.set_title(f"FoS over time — {Iz_fixed} mm/h rainfall", fontsize=11)
ax3.set_xlim(0, 48)
ax3.set_ylim(0, 2.5)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2, lw=0.5)

ax4 = axes2[1]
for (label, m0_val), col in zip(m0_scenarios.items(), colors_m0):
    m_t = mod.compute_m(const.T_HOURS, m0=m0_val, iz_mmh=Iz_fixed)
    ax4.plot(const.T_HOURS, m_t, color=col, lw=2.0, label=label)

ax4.axhline(
    1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="m=1.0 (full saturation)"
)
if 0 < m_star < 1:
    ax4.axhline(
        m_star,
        color="#A32D2D",
        lw=1.0,
        ls="-.",
        alpha=0.7,
        label=f"m*={m_star:.2f} (critical)",
    )

ax4.set_xlabel("Time since rainfall onset (hours)", fontsize=11)
ax4.set_ylabel("Saturation ratio m(t)", fontsize=11)
ax4.set_title(f"Saturation ratio — {Iz_fixed} mm/h rainfall", fontsize=11)
ax4.set_xlim(0, 48)
ax4.set_ylim(0, 1.05)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2, lw=0.5)

plt.tight_layout()
plt.savefig("fig2_antecedent_saturation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Saved: fig1_rainfall_intensity.png, fig2_antecedent_saturation.png")
