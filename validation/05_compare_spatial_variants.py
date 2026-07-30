"""
03_spatial_auc_validation.py - Consolidated Spatial & Temporal Validation

This script evaluates the model's predictive power across four different slope
configurations using a Case-Control validation algorithm.

To maximize computational efficiency, the hydrological bucket model is run only ONCE
per coordinate/date. The resulting pore-pressure profile is then simultaneously
evaluated against four terrain variants:
    1. Fixed 33 deg Baseline
    2. Single Map Pixel (exact coordinate)
    3. Map 3x3 Window (local max)
    4. Uphill Max Search (R=200 m, identifying the initiation scar)
"""

import sys
import os
import io
import contextlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import physics
from core import constants as const
from core import utils as ut
from validation import val_constants as auct
from validation import val_utils as autils

OUTPUT_DIR = f"{auct.OUTDIR}/05_spatial_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Raster Helper Functions ---


def assert_grids_compatible(dem_raster, slope_raster, tol=1e-3):
    """Fail loudly if the DEM and slope grids are not pixel-aligned.

    The uphill search combines DEM-derived and slope-derived arrays cell-by-cell
    through a SINGLE pixel window. That is only valid when both rasters share the
    same CRS, resolution and origin (identical affine transform). If they differ,
    the slope window would describe different ground than the DEM window and the
    uphill angle would be silently wrong — so we refuse to run instead.
    """
    if dem_raster.crs != slope_raster.crs:
        raise ValueError(
            f"CRS mismatch: DEM {dem_raster.crs} vs slope {slope_raster.crs}."
        )
    if not np.allclose(dem_raster.res, slope_raster.res, atol=tol):
        raise ValueError(
            f"Resolution mismatch: DEM {dem_raster.res} vs slope {slope_raster.res}. "
            "Point DEM_TIF at the SAME 25 m DEM the slope was derived from."
        )
    if not np.allclose(
        np.array(dem_raster.transform)[:6],
        np.array(slope_raster.transform)[:6],
        atol=tol,
    ):
        raise ValueError(
            "Grid origin mismatch: DEM and slope affine transforms differ. "
            "They must be pixel-aligned for the uphill search to be valid."
        )


def clean_raster_data(data, nodata_val):
    """Convert raster NoData sentinels to np.nan for safe math operations."""
    data = data.astype(float)
    if nodata_val is not None:
        data[data == nodata_val] = np.nan
    return data


def get_single_pixel_slope(slope_raster, x, y):
    """Extract the exact slope angle at the given LV95 coordinate.

    Deliberately NOT clamped to [15, 45]: a flat deposit pixel should stay flat
    so its (correct) high-FoS / "looks stable" behaviour shows up in the AUC.
    """
    row, col = slope_raster.index(x, y)
    data = clean_raster_data(
        slope_raster.read(
            1, window=Window(col, row, 1, 1), boundless=True, fill_value=np.nan
        ),
        slope_raster.nodata,
    )
    return float(data[0, 0]) if data.size > 0 else np.nan


def get_max_3x3_slope(slope_raster, x, y):
    """Steepest valid slope in a 3x3 window (robust to ~1-pixel coordinate error).

    Capped to the infinite-slope band [15, 45] so the max cannot latch onto a
    cliff pixel and break the physical assumption.
    """
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


def get_uphill_max_slope(dem_raster, slope_raster, x, y, radius_m=200):
    """Search uphill within a true circular radius to find the initiation scar.

    Candidate pixels must be (a) within radius_m of the coordinate, (b) strictly
    higher than the recorded deposit location, and (c) inside the [15, 45] band.
    Returns the steepest such slope, or NaN if none qualify.

    Grid note: assert_grids_compatible() guarantees the DEM and slope share one
    grid, so a single pixel window indexes the same ground in both rasters.
    """
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

    # 1. Circular radius mask (in metres, not pixels)
    yy, xx = np.ogrid[:win_size, :win_size]
    dist_sq = ((xx - r_pix) * pixel_size) ** 2 + ((yy - r_pix) * pixel_size) ** 2
    within_radius = dist_sq <= radius_m**2

    # 2. Uphill + validity-band masks
    uphill = dem_data > center_elev
    in_band = (slope_data >= auct.BETA_MIN) & (slope_data <= auct.BETA_MAX)

    candidates = slope_data[within_radius & uphill & in_band & np.isfinite(slope_data)]
    return float(candidates.max()) if candidates.size > 0 else np.nan


# --- Core Physics Evaluation ---


def get_fos_variants(x, y, date, dem_raster, slope_raster):
    """Run the bucket model once, then the minimum FoS for all four variants."""
    x, y = ut.to_lv95(x, y)
    region_id, drainage, et = dl.get_region_params(x, y, auct.CALIB)

    if drainage is None:
        return None

    # 1. Extract the four spatial angles
    angles = {
        "Fixed 33": const.BETA_DEG,
        "Single Pixel": get_single_pixel_slope(slope_raster, x, y),
        "Max 3x3": get_max_3x3_slope(slope_raster, x, y),
        "Uphill Max": get_uphill_max_slope(
            dem_raster, slope_raster, x, y, radius_m=200
        ),
    }

    # 2. Load rainfall and run the bucket model (the expensive part, done once!)
    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)

    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
    if rain is None or rain.loc[start:end].empty:
        return None

    rain = rain.loc[start:end]

    with contextlib.redirect_stdout(io.StringIO()):
        S = physics.calculate_daily_saturation(
            rain.values,
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
        )

    m_pp = physics.pore_pressure_ratio(S, const.S_PP_ONSET_DEFAULT)

    # 3. Calculate the minimum FoS in the window for each angle
    fos_results = {}
    for name, angle_deg in angles.items():
        if np.isnan(angle_deg) or angle_deg <= 0:
            fos_results[name] = np.nan
            continue

        beta_rad = np.radians(angle_deg)
        # Per-angle failure depth: keeps FoS correct once cohesion c > 0 is swept.
        h_v_dynamic = const.H_PERP / np.cos(beta_rad)

        fos = pd.Series(
            physics.compute_fos(
                m_array=m_pp,
                c=const.C,
                gamma=const.GAMMA,
                gamma_w=const.GAMMA_W,
                h_v=h_v_dynamic,
                beta_rad=beta_rad,
                phi_rad=const.phi,
            ),
            index=rain.index,
        )

        win = fos.loc[
            date
            - pd.Timedelta(days=auct.WINDOW_DAYS) : date
            + pd.Timedelta(days=auct.WINDOW_DAYS)
        ]
        fos_results[name] = float(win.min()) if not win.empty else np.nan

    return fos_results


# --- Main Execution ---


def main():
    print("Loading datasets and rasters...")
    inv = dl.load_wsl_usable_inventory()

    # For a quick smoke test, uncomment:
    # inv = inv.sample(min(150, len(inv)), random_state=42).reset_index(drop=True)

    with rasterio.open(auct.DEM_TIF) as dem_raster, rasterio.open(
        auct.SLOPE_TIF
    ) as slope_raster:

        # Refuse to run on misaligned grids (see function docstring).
        assert_grids_compatible(dem_raster, slope_raster)

        results = {
            "Fixed 33": {"pos": [], "neg": []},
            "Single Pixel": {"pos": [], "neg": []},
            "Max 3x3": {"pos": [], "neg": []},
            "Uphill Max": {"pos": [], "neg": []},
        }

        print(f"Evaluating {len(inv)} events...")
        for i, ev in inv.iterrows():

            fos_pos = get_fos_variants(
                ev["x"], ev["y"], ev["date"], dem_raster, slope_raster
            )
            if fos_pos is None:
                continue

            controls = list(autils.control_dates(ev["date"], auct.CONTROLS_PER_EVENT))
            fos_negs = [
                get_fos_variants(ev["x"], ev["y"], c_date, dem_raster, slope_raster)
                for c_date in controls
            ]

            if any(n is None for n in fos_negs):
                continue

            # COMMON SUBSET ENFORCEMENT:
            # Keep this event only if ALL four variants are finite for the event
            # AND for every one of its controls. Guarantees identical point sets.
            pos_valid = all(np.isfinite(fos_pos.get(v, np.nan)) for v in results)
            negs_valid = all(
                all(np.isfinite(n.get(v, np.nan)) for v in results) for n in fos_negs
            )

            if pos_valid and negs_valid:
                for variant in results:
                    results[variant]["pos"].append(fos_pos[variant])
                    for n in fos_negs:
                        results[variant]["neg"].append(n[variant])

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(inv)} events...")

    # --- Statistical Output & Visualization ---
    plt.figure(figsize=(10, 8))

    print("\n--- Final Spatial AUC Results (Common Valid Subset) ---")
    colors = {
        "Fixed 33": "gray",
        "Single Pixel": "blue",
        "Max 3x3": "green",
        "Uphill Max": "firebrick",
    }

    for variant, data in results.items():
        pos = np.array(data["pos"])
        neg = np.array(data["neg"])

        if len(pos) == 0:
            print(f"{variant}: Insufficient data.")
            continue

        auc = autils.auc_score(pos, neg)
        print(f"{variant:<15} | AUC: {auc:.3f} | Valid Shared Events: {len(pos)}")

        thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
        tpr = [(pos <= t).mean() for t in thr]
        fpr = [(neg <= t).mean() for t in thr]

        plt.plot(
            fpr,
            tpr,
            label=f"{variant} (AUC = {auc:.3f})",
            color=colors[variant],
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")

    plt.title(
        "ROC/AUC Comparison of Spatial Integration Variants\n"
        "(Evaluated on Shared Valid Subset)",
        fontsize=14,
    )
    plt.xlabel("False Positive Rate (Control Days Triggered)", fontsize=12)
    plt.ylabel("True Positive Rate (Disaster Days Triggered)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    output_path = f"{OUTPUT_DIR}/spatial_roc_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved combined ROC plot to: {output_path}")


if __name__ == "__main__":
    main()
