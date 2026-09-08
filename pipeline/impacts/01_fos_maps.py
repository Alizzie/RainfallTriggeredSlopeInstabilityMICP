"""
01_fos_maps.py - National FoS maps: bare soil, roots, MICP, and roots + MICP.
Design saturation is the worst case (m_pp=1.0, fixed).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio

from core import physics
from core import constants as const
from core import data_loader as dl
from pipeline.utils import constants as auct

NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"

ROOT_C = {
    12: 15.0,  # closed forest
    14: 10.0,  # dense small stands (copse)
    13: 8.0,  # open forest
    6: 6.0,  # bush forest
}

# literature sensitivity envelope around ROOT_C ("mid"), roughly +-40%.
ROOT_C_SWEEP = {
    "low": {12: 8.0, 14: 6.0, 13: 5.0, 6: 5.0},
    "mid": ROOT_C,
    "high": {12: 22.0, 14: 15.0, 13: 12.0, 6: 8.0},
}

MICP_C = 2.0

M_PP_DESIGN = 0.12  # worst case
BETA_MIN, BETA_MAX = auct.BETA_MIN, auct.BETA_MAX
BARE_C = const.C  # calibrated in soil_cohesion.py (~0.0 kPa)

OUTDIR = f"output/impacts/01_maps/m_pp_{M_PP_DESIGN}"
os.makedirs(OUTDIR, exist_ok=True)


def fos_field(beta_deg, m_pp, c_field):
    """FoS for a given slope angle, pore pressure, and cohesion field."""
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


def required_cohesion(beta_deg, m_pp):
    """Cohesion [kPa] needed to exactly reach FoS = 1 at this pore-pressure ratio."""
    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)
    tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    fos0 = physics.compute_fos(
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta_rad,
        phi_rad=const.phi,
        m_array=m_pp,
    )
    return np.maximum((1.0 - fos0) * tau, 0.0)


def read_aligned(path, ref_shape, name):
    """Read a raster and check it matches a reference shape."""
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != ref_shape:
        raise ValueError(
            f"{name} shape {a.shape} does not match reference shape {ref_shape}"
        )
    return a


def cohesion_from(forest, mapping):
    """Root cohesion field from a forest classification raster and a class->kPa mapping."""
    c = np.zeros_like(forest, dtype=np.float32)
    for cls, val in mapping.items():
        c[forest == cls] = val
    return c


def write(beta, valid, profile, name, field, dtype=np.float32, nodata=np.nan):
    """Write one field to OUTDIR/name.tif, filling nodata outside `valid`."""
    out = np.full(beta.shape, nodata, dtype=dtype)
    out[valid] = field
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata)
    with rasterio.open(f"{OUTDIR}/{name}_{M_PP_DESIGN}.tif", "w", **p) as dst:
        dst.write(out, 1)


def main():
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta = src.read(1).astype(np.float32)
        profile = src.profile
        slope_nd = src.nodata

    forest, _, _ = dl.load_forest_raster()
    if forest.shape != beta.shape:
        raise ValueError(f"forest shape {forest.shape} != slope shape {beta.shape}")
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
        f"({1 - n_final / max(n_slope, 1):.1%})"
    )
    print(f"  soil-mantled pixels used: {n_final:,}")

    b = beta[valid]
    f = forest[valid]

    c_root = cohesion_from(f, ROOT_C)
    c_micp = np.full(b.shape, MICP_C, dtype=np.float32)

    fos_bare = fos_field(b, M_PP_DESIGN, BARE_C)
    fos_root = fos_field(b, M_PP_DESIGN, BARE_C + c_root)
    fos_micp = fos_field(b, M_PP_DESIGN, BARE_C + c_micp)
    fos_root_micp = fos_field(b, M_PP_DESIGN, BARE_C + c_root + c_micp)

    c_req = required_cohesion(b, M_PP_DESIGN)
    r_req_root = np.maximum(c_req - c_root, 0.0)
    c_req_micp = np.maximum(c_req - c_micp, 0.0)
    c_req_root_micp = np.maximum(c_req - c_root - c_micp, 0.0)

    write(beta, valid, profile, "fos_bare", fos_bare)
    write(beta, valid, profile, "fos_root", fos_root)
    write(beta, valid, profile, "fos_micp", fos_micp)
    write(beta, valid, profile, "fos_root_micp", fos_root_micp)

    write(beta, valid, profile, "required_cohesion", c_req)
    write(beta, valid, profile, "required_cohesion_above_roots", r_req_root)
    write(beta, valid, profile, "required_cohesion_above_micp", c_req_micp)
    write(
        beta, valid, profile, "required_cohesion_above_roots_and_micp", c_req_root_micp
    )

    unstable_bare = fos_bare < 1
    unstable_root = fos_root < 1
    unstable_micp = fos_micp < 1
    unstable_root_micp = fos_root_micp < 1

    print(f"forested share of soil pixels: {(c_root > 0).mean():.1%}")
    print(f"MICP additional cohesion: {MICP_C:.2f} kPa")

    print(f"unstable (FoS < 1) at m_pp={M_PP_DESIGN:g}, c_bare={BARE_C:.2f} kPa:")
    print(f"bare soil               : {unstable_bare.mean():.1%}")
    print(f"+ roots                 : {unstable_root.mean():.1%}")
    print(f"+ MICP ({MICP_C:g} kPa)       : {unstable_micp.mean():.1%}")
    print(f"+ roots + MICP          : {unstable_root_micp.mean():.1%}")

    print("")
    print("share of bare-unstable pixels stabilized:")
    print(
        f"roots        : {(unstable_bare & ~unstable_root).sum() / max(unstable_bare.sum(), 1):.1%}"
    )
    print(
        f"MICP         : {(unstable_bare & ~unstable_micp).sum() / max(unstable_bare.sum(), 1):.1%}"
    )
    print(
        f"roots + MICP : {(unstable_bare & ~unstable_root_micp).sum() / max(unstable_bare.sum(), 1):.1%}"
    )

    print("")
    print(
        f"required cohesion: median={np.median(c_req):.2f}, p90={np.percentile(c_req, 90):.2f}, max={c_req.max():.2f} kPa"
    )

    print("")
    print("sensitivity to the root-cohesion assumption:")
    print(f"{'variant':>8} {'unstable_root':>15} {'unstable_root+micp':>20}")

    for name, mapping in ROOT_C_SWEEP.items():
        cr = cohesion_from(f, mapping)
        fos_r = fos_field(b, M_PP_DESIGN, BARE_C + cr)
        fos_rm = fos_field(b, M_PP_DESIGN, BARE_C + cr + c_micp)
        print(f"{name:>8} {(fos_r < 1).mean():14.1%} {(fos_rm < 1).mean():19.1%}")

    print("")
    print(f"results saved to {OUTDIR}")


if __name__ == "__main__":
    main()
