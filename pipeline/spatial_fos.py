"""
spatial_fos.py — National shallow-landslide FoS map, with and without MICP.

Turns the point/temporal engine (model.compute_fos) into a spatial map by feeding
a per-pixel slope angle (the QGIS slope raster) instead of the fixed BETA_DEG.

compute_fos already broadcasts over arrays, so it is reused UNCHANGED. The only
adjustment vs. the temporal code is that the failure depth H_v must be derived
per pixel from the local slope (H_PERP / cos beta), not from the scalar const.H_V.

Outputs:
  fos_baseline.tif  — FoS at the design wetness, current cohesion
  fos_micp.tif      — FoS at the design wetness, cohesion + MICP gain
  fos_rescued.tif   — 1 where baseline < 1 AND micp >= 1  (where MICP helps)

MEMORY: run the NATIONAL map on a resampled (25 m for 16 GB RAM, 50 m for 8 GB)
slope raster. Native 10 m over all of Switzerland is ~2.4e9 pixels ≈ 9.5 GB per
float32 array and will not fit. Use native 10 m only on the clipped target sites.
"""

import numpy as np
import rasterio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import physics
from core import constants as const

# --- inputs / outputs ---
SLOPE_TIF = "data/swissalti_slope/slope_deg_10m.tif"  # QGIS: Raster > Analysis > Slope (Degrees)
OUT_BASELINE = "output/fos_baseline_10m.tif"
OUT_MICP = "output/fos_micp_10m.tif"
OUT_RESCUED = "output/fos_rescued_10m.tif"

# --- scenario knobs ---
# Design pore-pressure driver m_pp in [0, 1]: 0 = dry, 1 = saturated above onset.
# Spatial equivalent of picking a rainfall scenario. Sweep 0.3 / 0.6 / 1.0 for a
# "dry / wet / design-storm" set of maps, mirroring RAIN_SCENARIOS in the time model.
M_PP_DESIGN = 1.0

MICP_COHESION_GAIN = 15.0  # kPa added by treatment (as in hist_simulation)

# Infinite-slope theory only holds on soil-mantled slopes. Mask the rest so you
# don't map FoS on flats (tau -> 0, FoS -> inf) or rock walls (too steep).
BETA_MIN_DEG = 15.0
BETA_MAX_DEG = 45.0


def fos_field(beta_deg, m_pp, cohesion):
    """Vectorised FoS over a slope field. compute_fos is reused as-is."""
    beta = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(
        beta
    )  # per-pixel failure depth (replaces scalar const.H_V)
    return physics.compute_fos(
        c=cohesion,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta,
        phi_rad=const.phi,
        m_array=m_pp,  # scalar scenario (or a 2-D field, if you have one)
    )


def main():
    with rasterio.open(SLOPE_TIF) as src:
        beta_deg = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata

    # valid soil-slope mask
    valid = np.isfinite(beta_deg)
    if nodata is not None:
        valid &= beta_deg != nodata
    valid &= (beta_deg >= BETA_MIN_DEG) & (beta_deg <= BETA_MAX_DEG)

    baseline = np.full(beta_deg.shape, np.nan, dtype="float32")
    micp = np.full(beta_deg.shape, np.nan, dtype="float32")

    # compute only on valid pixels (1-D views -> cheap, avoids rock/flat artefacts)
    baseline[valid] = fos_field(beta_deg[valid], M_PP_DESIGN, const.C)
    micp[valid] = fos_field(beta_deg[valid], M_PP_DESIGN, const.C + MICP_COHESION_GAIN)

    # where treatment lifts an unstable pixel to stable — the money map
    rescued = np.where(valid & (baseline < 1.0) & (micp >= 1.0), 1, 0).astype("uint8")

    profile.update(dtype="float32", count=1, nodata=np.nan)
    for path, arr in [(OUT_BASELINE, baseline), (OUT_MICP, micp)]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    prof_u8 = profile.copy()
    prof_u8.update(dtype="uint8", nodata=0)
    with rasterio.open(OUT_RESCUED, "w", **prof_u8) as dst:
        dst.write(rescued, 1)

    n = int(valid.sum())
    unstable = int((baseline[valid] < 1.0).sum())
    saved = int(rescued.sum())
    print(f"design m_pp = {M_PP_DESIGN}, MICP gain = +{MICP_COHESION_GAIN} kPa")
    print(f"valid soil pixels ({BETA_MIN_DEG:.0f}-{BETA_MAX_DEG:.0f} deg): {n:,}")
    print(f"unstable baseline (FoS<1):   {unstable:,} ({unstable / n:.1%})")
    print(
        f"rescued by MICP:             {saved:,} "
        f"({saved / n:.1%} of valid, {saved / max(unstable, 1):.1%} of unstable)"
    )
    print(f"-> {OUT_BASELINE} | {OUT_MICP} | {OUT_RESCUED}")


if __name__ == "__main__":
    main()


# DEM 25m:

# design m_pp = 1.0, MICP gain = +15.0 kPa
# valid soil pixels (15-45 deg): 118,688,990
# unstable baseline (FoS<1):   49,104,744 (41.4%)
# rescued by MICP:             49,104,744 (41.4% of valid, 100.0% of unstable)
# -> output/fos_baseline.tif | output/fos_micp.tif | output/fos_rescued.tif


# DEM 10m:
# design m_pp = 1.0, MICP gain = +15.0 kPa
# valid soil pixels (15-45 deg): 749,009,745
# unstable baseline (FoS<1):   324,019,320 (43.3%)
# rescued by MICP:             324,019,320 (43.3% of valid, 100.0% of unstable)
# -> output/fos_baseline_10m.tif | output/fos_micp_10m.tif | output/fos_rescued_10m.tif

# => DEM 10m is more accurate, and shows more unstable pixels, but the 25m DEM is still useful for national-scale mapping and is much faster to compute.
