"""
cohesion_vs_slope_variant.py - Does cohesion help more with REAL per-pixel
slope angles than with one fixed global angle?

Output (in OUTPUT_DIR):
    cohesion_by_variant.csv        AUC / J / MCC per (variant, cohesion) cell
    cohesion_by_variant.png        three panels (AUC, J, MCC) vs cohesion,
                                    one line per variant
"""

import sys
import os
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

OUTPUT_DIR = "output/spatial/05_soil_cohesion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_LABEL = f"Fixed {const.BETA_DEG}\u00b0"
VARIANT_NAMES = [FIXED_LABEL, "Single Pixel", "Max 3x3", "Uphill Max"]
COHESION_GRID_KPA = np.arange(0.0, 12.01, 0.25)
COLORS = {
    FIXED_LABEL: "gray",
    "Single Pixel": "blue",
    "Max 3x3": "green",
    "Uphill Max": "firebrick",
}


def resolve_uphill_radius(default=60):
    """Same as 04_compare_spatial_variants.py: pick the search radius from
    03_sweep_uphill_radius.py's hydro-only AUC result."""
    csv_path = "output/spatial/03_uphill_sweep/auc_vs_radius.csv"
    if not os.path.exists(csv_path):
        print(f"[warn] {csv_path} not found -- using default radius={default} m.")
        return default
    df = pd.read_csv(csv_path)
    best = df.loc[df["auc_hydro"].idxmax()]
    return int(best["radius_m"])


UPHILL_RADIUS_M = resolve_uphill_radius()


# --- Raster helpers (copied from 04_compare_spatial_variants.py) ---


def clean_raster_data(data, nodata_val):
    data = data.astype(float)
    if nodata_val is not None:
        data[data == nodata_val] = np.nan
    return data


def get_single_pixel_slope(slope_raster, x, y):
    row, col = slope_raster.index(x, y)
    data = clean_raster_data(
        slope_raster.read(
            1, window=Window(col, row, 1, 1), boundless=True, fill_value=np.nan
        ),
        slope_raster.nodata,
    )
    return float(data[0, 0]) if data.size > 0 else np.nan


def get_max_3x3_slope(slope_raster, x, y):
    row, col = slope_raster.index(x, y)
    data = clean_raster_data(
        slope_raster.read(
            1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan
        ),
        slope_raster.nodata,
    )
    in_band = (data >= auct.BETA_MIN) & (data <= auct.BETA_MAX) & np.isfinite(data)
    candidates = data[in_band]
    return float(candidates.max()) if candidates.size > 0 else np.nan


def get_uphill_max_slope(dem_raster, slope_raster, x, y, radius_m):
    row, col = dem_raster.index(x, y)
    pixel_size = dem_raster.res[0]
    r_pix = int(round(radius_m / pixel_size))
    win_size = 2 * r_pix + 1
    window = Window(col - r_pix, row - r_pix, win_size, win_size)

    dem_data = clean_raster_data(
        dem_raster.read(1, window=window, boundless=True, fill_value=np.nan),
        dem_raster.nodata,
    )
    slope_data = clean_raster_data(
        slope_raster.read(1, window=window, boundless=True, fill_value=np.nan),
        slope_raster.nodata,
    )
    center_elev = dem_data[r_pix, r_pix]
    if not np.isfinite(center_elev):
        return np.nan

    yy, xx = np.ogrid[:win_size, :win_size]
    dist_sq = ((xx - r_pix) * pixel_size) ** 2 + ((yy - r_pix) * pixel_size) ** 2
    within_radius = dist_sq <= radius_m**2
    uphill = dem_data > center_elev
    in_band = (slope_data >= auct.BETA_MIN) & (slope_data <= auct.BETA_MAX)

    candidates = slope_data[within_radius & uphill & in_band & np.isfinite(slope_data)]
    return float(candidates.max()) if candidates.size > 0 else np.nan


def assert_grids_compatible(dem_raster, slope_raster, tol=1e-3):
    if dem_raster.crs != slope_raster.crs:
        raise ValueError("CRS mismatch between DEM and slope rasters.")
    if not np.allclose(dem_raster.res, slope_raster.res, atol=tol):
        raise ValueError("Resolution mismatch between DEM and slope rasters.")
    if not np.allclose(
        np.array(dem_raster.transform)[:6],
        np.array(slope_raster.transform)[:6],
        atol=tol,
    ):
        raise ValueError("Grid origin mismatch between DEM and slope rasters.")


# --- Pore pressure (same closed form as soil_cohesion.py) ---


def max_pore_pressure_ratio(x, y, date):
    """Max pore-pressure ratio in the +/- window around date, or NaN if no rainfall data."""
    x_lv95, y_lv95 = dl.to_lv95(x, y)
    _, drainage, et, et_amp = dl.get_region_params(x_lv95, y_lv95, auct.CALIB)
    if drainage is None:
        return np.nan

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x_lv95, y_lv95, sorted({start.year, end.year}))
    if rain is None:
        return np.nan
    rain = rain.loc[start:end]
    if rain.empty:
        return np.nan

    with contextlib.redirect_stdout(io.StringIO()):
        saturation = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
            et_amplitude=et_amp,
            day_of_year=rain.index.dayofyear.to_numpy(),
        )
    pore_pressure = pd.Series(
        physics.pore_pressure_ratio(saturation, const.S_PP_ONSET_DEFAULT),
        index=rain.index,
    )
    window = pore_pressure.loc[
        date
        - pd.Timedelta(days=auct.WINDOW_DAYS) : date
        + pd.Timedelta(days=auct.WINDOW_DAYS)
    ]
    return float(window.max()) if not window.empty else np.nan


def fos0_and_tau(pore_pressure_ratios, beta_deg):
    """(FoS at c'=0, tau) for arrays of pore-pressure ratios AND slope angles.

    Any cohesion's FoS follows as fos_at_c0 + cohesion / tau, evaluated
    elementwise -- both inputs can be per-point arrays here, unlike
    soil_cohesion.py where beta was one shared value for every point.
    """
    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)
    tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    fos_at_c0 = physics.compute_fos(
        m_array=pore_pressure_ratios,
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta_rad,
        phi_rad=const.phi,
    )
    return fos_at_c0, tau


# --- Main data collection ---


def collect_case_control_points(dem_raster, slope_raster):
    """One row per event and one per control, kept only if EVERY variant's
    angle is in [15, 45] deg (see module docstring for why the full band).

    Returns a dict {variant_name: {"event": [...], "control": [...]}} of
    (pore_pressure_ratio, beta_deg) pairs.
    """
    inventory = dl.load_wsl_usable_inventory()
    points = {v: {"event": [], "control": []} for v in VARIANT_NAMES}
    n_candidates = n_kept = 0

    for i, event in inventory.iterrows():
        x, y = event["x"], event["y"]
        x_lv95, y_lv95 = dl.to_lv95(x, y)

        angles = {
            FIXED_LABEL: float(const.BETA_DEG),
            "Single Pixel": get_single_pixel_slope(slope_raster, x_lv95, y_lv95),
            "Max 3x3": get_max_3x3_slope(slope_raster, x_lv95, y_lv95),
            "Uphill Max": get_uphill_max_slope(
                dem_raster, slope_raster, x_lv95, y_lv95, radius_m=UPHILL_RADIUS_M
            ),
        }
        regimes = {v: autils.classify_slope(a) for v, a in angles.items()}
        angle_ok = all(regimes[v] != "invalid" for v in VARIANT_NAMES)
        if not angle_ok:
            continue

        event_pp = max_pore_pressure_ratio(x, y, event["date"])
        controls = vm.find_control_date(
            x, y, event["date"], inventory, n=auct.CONTROLS_PER_EVENT
        )
        control_pps = [max_pore_pressure_ratio(x, y, c) for c in controls]
        control_pps = [m for m in control_pps if np.isfinite(m)]

        n_candidates += 1
        if not (np.isfinite(event_pp) and control_pps):
            continue

        n_kept += 1
        for v in VARIANT_NAMES:
            points[v]["event"].append((event_pp, angles[v]))
            for m in control_pps:
                points[v]["control"].append((m, angles[v]))

        if (i + 1) % 50 == 0:
            print(
                f"  {i + 1}/{len(inventory)} processed (kept {n_kept}/{n_candidates})"
            )

    print(f"\n{n_kept}/{n_candidates} candidate events kept (full [15, 45] deg band)")
    return points


def score_variant_across_cohesion(event_points, control_points):
    """AUC / J / MCC at every cohesion value, for one variant's points."""
    event_pp, event_beta = map(np.array, zip(*event_points))
    control_pp, control_beta = map(np.array, zip(*control_points))

    fos0_events, tau_events = fos0_and_tau(event_pp, event_beta)
    fos0_controls, tau_controls = fos0_and_tau(control_pp, control_beta)

    rows = []
    for cohesion in COHESION_GRID_KPA:
        fos_events = fos0_events + cohesion / tau_events
        fos_controls = fos0_controls + cohesion / tau_controls
        event_flags = fos_events <= 1.0
        control_flags = fos_controls <= 1.0

        auc = autils.auc_score(fos_events, fos_controls)
        tpr, fpr = vm.tpr_fpr(event_flags, control_flags)
        rows.append(
            {
                "cohesion_kpa": cohesion,
                "auc": auc,
                "youden_j": vm.youden_j(event_flags, control_flags),
                "mcc": vm.mcc(event_flags, control_flags),
                "tpr": tpr,
                "fpr": fpr,
                "n": len(event_pp),
            }
        )
    return pd.DataFrame(rows)


def plot_metrics_by_variant(all_curves):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for variant, curve in all_curves.items():
        color = COLORS[variant]
        axes[0].plot(curve.cohesion_kpa, curve.auc, color=color, lw=2, label=variant)
        axes[1].plot(
            curve.cohesion_kpa, curve.youden_j, color=color, lw=2, label=variant
        )
        axes[2].plot(curve.cohesion_kpa, curve.mcc, color=color, lw=2, label=variant)

    for ax, title in zip(axes, ["AUC", "Youden's J", "MCC"]):
        ax.set_xlabel("cohesion c' (kPa)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Cohesion's effect per slope-sampling variant (full 15-45\u00b0 band)", y=1.02
    )
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/cohesion_by_variant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    with rasterio.open(auct.DEM_TIF) as dem_raster, rasterio.open(
        auct.SLOPE_TIF
    ) as slope_raster:
        assert_grids_compatible(dem_raster, slope_raster)
        print("Collecting event/control points and slope angles...")
        points = collect_case_control_points(dem_raster, slope_raster)

    print(
        f"\nScoring {len(COHESION_GRID_KPA)} cohesion values x {len(VARIANT_NAMES)} variants..."
    )
    all_curves = {}
    all_rows = []
    for variant in VARIANT_NAMES:
        curve = score_variant_across_cohesion(
            points[variant]["event"], points[variant]["control"]
        )
        all_curves[variant] = curve
        curve = curve.assign(variant=variant)
        all_rows.append(curve)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/cohesion_by_variant.csv", index=False)
    plot_metrics_by_variant(all_curves)

    print(
        f"\n{'variant':>15} | {'AUC range':>18} | {'best J':>8} | {'at cohesion':>12}"
    )
    for variant, curve in all_curves.items():
        auc_range = f"{curve.auc.min():.3f} - {curve.auc.max():.3f}"
        best = curve.loc[curve["youden_j"].idxmax()]
        print(
            f"{variant:>15} | {auc_range:>18} | {best.youden_j:8.3f} | {best.cohesion_kpa:12.2f}"
        )

    print(
        f"\nAUC held constant (a flat range above) means cohesion only moved the "
        f"threshold for that variant, same as the global-beta result in "
        f"soil_cohesion.py. AUC that actually moves with cohesion means real "
        f"per-pixel angle + cohesion is doing something a global angle can't."
    )
    print(f"\n-> {OUTPUT_DIR}/ (cohesion_by_variant.csv, cohesion_by_variant.png)")


if __name__ == "__main__":
    main()
