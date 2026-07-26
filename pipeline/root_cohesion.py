"""
SwissTLM3D land cover with root cohesion approximation and a bedrock mask.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio

from core import physics
from core import constants as const

SLOPE_TIF = "data/swissalti_slope/slope_deg_25m_ch.tif"
FOREST_TIF = "data/swissTLM3D/tlm_forest_25m_ch.tif"
NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"
OUTDIR = "output/root_cohesion"
os.makedirs(OUTDIR, exist_ok=True)

ROOT_C = {
    12: 15.0,  # closed forest
    14: 10.0,  # (dense small stands copse)
    13: 8.0,  # (open forest)
    6: 6.0,  # (bush forest)
}

ROOT_C_SWEEP = {
    "low": {12: 8.0, 14: 6.0, 13: 5.0, 6: 5.0},
    "mid": ROOT_C,
    "high": {12: 22.0, 14: 15.0, 13: 12.0, 6: 8.0},
}

MICP_GAIN = 15.0
M_PP_DESIGN = 1.0  # worst case
BETA_MIN, BETA_MAX = 15.0, 45.0
BARE_C = 0.25


def fos_field(beta_deg, m_pp, c_field):
    """Compute FoS for a given slope angle, pore pressure, and root cohesion field."""
    beta_rad = np.radians(beta_deg)
    return physics.compute_fos(
        c=c_field,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        beta_rad=beta_rad,
        h_v=const.H_PERP / np.cos(beta_rad),
        m_array=m_pp,
        phi_rad=const.phi,
    )


def read_aligned(path, ref_shape, name):
    """Read a raster and check that it has the same shape as a reference shape."""
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != ref_shape:
        raise ValueError(
            f"{name} shape {a.shape} does not match reference shape {ref_shape}"
        )
    return a


def cohesion_from(forest, mapping):
    """Create a root cohesion field from a forest classification raster and a mapping of classes to cohesion values."""
    c = np.zeros_like(forest, dtype=np.float32)
    for cls, val in mapping.items():
        c[forest == cls] = val
    return c


def write(beta, valid, profile, name, fos_field, dtype=np.float32, nodata=np.nan):
    """Write a raster to the output directory with the given name and data type."""
    out = np.full(beta.shape, nodata, dtype=dtype)
    out[valid] = fos_field
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata)
    with rasterio.open(f"{OUTDIR}/{name}.tif", "w", **p) as dst:
        dst.write(out, 1)


def main():
    with rasterio.open(SLOPE_TIF) as src:
        beta = src.read(1).astype(np.float32)
        profile = src.profile
        slope_nd = src.nodata

    forest = read_aligned(FOREST_TIF, beta.shape, "forest")
    nonsoil = read_aligned(NONSOIL_TIF, beta.shape, "non-soil")

    valid = np.isfinite(beta)
    if slope_nd is not None:
        valid &= beta != slope_nd

    valid &= (beta >= BETA_MIN) & (beta <= BETA_MAX)
    n_slope = int(valid.sum())
    valid &= nonsoil != 1
    n_final = int(valid.sum())
    print(f"pixels in {BETA_MIN:.0f}-{BETA_MAX:.0f} deg: {n_slope:,}")
    print(
        f"  removed as rock/glacier/water/snow: {n_slope - n_final:,} "
        f"({1 - n_final / max(n_slope,1):.1%})"
    )
    print(f"  soil-mantled pixels used: {n_final:,}")

    b = beta[valid]
    f = forest[valid]

    c_root = cohesion_from(f, ROOT_C)
    fos_bare = fos_field(b, M_PP_DESIGN, BARE_C)
    fos_root = fos_field(b, M_PP_DESIGN, BARE_C + c_root)
    fos_micp = fos_field(b, M_PP_DESIGN, BARE_C + c_root + MICP_GAIN)
    rescued = ((fos_root < 1.0) & (fos_micp >= 1.0)).astype("uint8")

    write(beta, valid, profile, "fos_bare", fos_bare)
    write(beta, valid, profile, "fos_root", fos_root)
    write(beta, valid, profile, "fos_root_micp", fos_micp)
    write(
        beta,
        valid,
        profile,
        "fos_rescued_by_micp",
        rescued,
        dtype="uint8",
        nodata=0,
    )

    print(f"\nforested share of soil pixels: {(c_root > 0).mean():.1%}")
    print(f"unstable (FoS<1) at m_pp={M_PP_DESIGN}:")
    print(f"  bare soil (c={BARE_C})   : {(fos_bare < 1).mean():.1%}")
    print(f"  + roots                : {(fos_root < 1).mean():.1%}   <- present day")
    print(f"  + roots + MICP         : {(fos_micp < 1).mean():.1%}")
    print(
        f"\nrescued by MICP vs present day: {rescued.sum():,} px "
        f"({rescued.mean():.1%} of soil pixels, "
        f"{rescued.sum() / max(int((fos_root < 1).sum()), 1):.1%} of currently unstable)"
    )

    print("\nsensitivity to the root-cohesion assumption:")
    print(f"{'variant':>8} {'unstable_now':>13} {'rescued_by_MICP':>16}")

    for name, mapping in ROOT_C_SWEEP.items():
        cr = cohesion_from(f, mapping)
        fos_r = fos_field(b, M_PP_DESIGN, BARE_C + cr)
        fos_m = fos_field(b, M_PP_DESIGN, BARE_C + cr + MICP_GAIN)
        print(
            f"{name:>8} {(fos_r < 1).mean():12.1%} {((fos_r < 1) & (fos_m >= 1)).mean():16.1%}"
        )

    print(
        f"\n-> {OUTDIR}/fos_bare.tif | fos_root.tif | fos_root_micp.tif | "
        f"fos_rescued_by_micp.tif"
    )


if __name__ == "__main__":
    main()
