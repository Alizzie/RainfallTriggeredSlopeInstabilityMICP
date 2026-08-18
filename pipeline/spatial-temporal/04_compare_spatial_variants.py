"""
04_compare_spatial_variants.py - Consolidated Spatial & Temporal Validation

This script evaluates the model's predictive power across four different slope
configurations using a Case-Control validation algorithm.

To maximize computational efficiency, the hydrological bucket model is run only ONCE
per coordinate/date. The resulting pore-pressure profile is then simultaneously
evaluated against four terrain variants:
    1. Fixed 30 deg Baseline
    2. Single Map Pixel (exact coordinate)
    3. Map 3x3 Window (local max)
    4. Uphill Max Search (radius picked from 03_sweep_uphill_radius.py's auc_hydro result)
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import data_loader as dl
from core import physics
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

OUTPUT_DIR = f"output/spatial/04_spatial_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_LABEL = f"Fixed {const.BETA_DEG}°"
MODES = ("hydro", "full")
MODE_LABELS = {"hydro": "15-35°", "full": "15-45°"}


def resolve_uphill_radius(default=60):
    """Pull the best search radius from 03_sweep_uphill_radius.py's output CSV"""
    csv_path = "output/spatial/03_uphill_sweep/auc_vs_radius.csv"
    if not os.path.exists(csv_path):
        print(
            f"[warn] {csv_path} not found -- using default radius={default} m. "
            f"Run 04_sweep_uphill_radius.py first to pick this from data."
        )
        return default
    df = pd.read_csv(csv_path)
    best = df.loc[df["auc_hydro"].idxmax()]
    r = int(best["radius_m"])
    print(
        f"Using uphill radius {r} m (Phase 04 result, AUC_hydro={best['auc_hydro']:.3f})"
    )
    return r


UPHILL_RADIUS_M = resolve_uphill_radius()

# --- Raster Helper Functions ---


def assert_grids_compatible(dem_raster, slope_raster, tol=1e-3):
    """Fail loudly if the DEM and slope grids are not pixel-aligned."""
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
    """Extract the exact slope angle at the given LV95 coordinate."""
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
    Candidate pixels must be inside the [15, 45] deg validity band. Returns NaN if none qualify.
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


def get_uphill_max_slope(dem_raster, slope_raster, x, y, radius_m):
    """Search uphill within a true circular radius to find the initiation scar."""
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
    """Run the bucket model once, then the minimum FoS + sampled angle for all
    four variants. Returns (fos_results, angles) or None if rainfall is missing.
    """
    x, y = dl.to_lv95(x, y)
    _, drainage, et, et_amp = dl.get_region_params(x, y, auct.CALIB)

    if drainage is None:
        return None

    # 1. Extract the four spatial angles
    angles = {
        FIXED_LABEL: const.BETA_DEG,
        "Single Pixel": get_single_pixel_slope(slope_raster, x, y),
        "Max 3x3": get_max_3x3_slope(slope_raster, x, y),
        "Uphill Max": get_uphill_max_slope(
            dem_raster, slope_raster, x, y, radius_m=UPHILL_RADIUS_M
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
            et_amplitude=et_amp,
            day_of_year=rain.index.dayofyear.to_numpy(),
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

    return fos_results, angles


# --- Main Execution ---


def main():
    print("Loading datasets and rasters...")
    inv = dl.load_wsl_usable_inventory()

    # For a quick smoke test, uncomment:
    # inv = inv.sample(min(150, len(inv)), random_state=42).reset_index(drop=True)

    variant_names = [FIXED_LABEL, "Single Pixel", "Max 3x3", "Uphill Max"]

    with rasterio.open(auct.DEM_TIF) as dem_raster, rasterio.open(
        auct.SLOPE_TIF
    ) as slope_raster:

        # Refuse to run on misaligned grids (see function docstring).
        assert_grids_compatible(dem_raster, slope_raster)

        results = {
            mode: {name: {"pos": [], "neg": []} for name in variant_names}
            for mode in MODES
        }
        regime_counts = {
            name: {"hydro": 0, "dry": 0, "invalid": 0} for name in variant_names
        }
        n_candidates = 0  # events with full data (before either gate)
        n_kept = {mode: 0 for mode in MODES}

        print(f"Evaluating {len(inv)} events...")
        for i, ev in inv.iterrows():

            out_pos = get_fos_variants(
                ev["x"], ev["y"], ev["date"], dem_raster, slope_raster
            )
            if out_pos is None:
                continue
            fos_pos, angles_pos = out_pos

            controls = vm.find_control_date(
                ev["x"], ev["y"], ev["date"], inv, n=auct.CONTROLS_PER_EVENT
            )
            outs_neg = [
                get_fos_variants(ev["x"], ev["y"], c_date, dem_raster, slope_raster)
                for c_date in controls
            ]

            if not controls or any(n is None for n in outs_neg):
                continue
            fos_negs = [n[0] for n in outs_neg]
            n_candidates += 1

            regimes = {v: autils.classify_slope(angles_pos[v]) for v in variant_names}
            for v in variant_names:
                regime_counts[v][regimes[v]] += 1

            pos_finite = all(np.isfinite(fos_pos.get(v, np.nan)) for v in variant_names)
            negs_finite = all(
                all(np.isfinite(n.get(v, np.nan)) for v in variant_names)
                for n in fos_negs
            )
            if not (pos_finite and negs_finite):
                continue

            for mode in MODES:
                if mode == "hydro":
                    angle_ok = all(regimes[v] == "hydro" for v in variant_names)
                else:  # "full": in-band (hydro or dry), just not "invalid"
                    angle_ok = all(regimes[v] != "invalid" for v in variant_names)
                if not angle_ok:
                    continue
                n_kept[mode] += 1
                for variant in variant_names:
                    results[mode][variant]["pos"].append(fos_pos[variant])
                    for n in fos_negs:
                        results[mode][variant]["neg"].append(n[variant])

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(inv)} events...")

    # --- Regime counts (composition of the full candidate pool, both gates draw from this) ---
    print(f"\n{'variant':>15}  {'hydro':>7} {'dry':>6} {'invalid':>8}")
    for variant in variant_names:
        c = regime_counts[variant]
        print(f"{variant:>15}  {c['hydro']:7d} {c['dry']:6d} {c['invalid']:8d}")
    for mode in MODES:
        print(
            f"{n_kept[mode]}/{n_candidates} candidate events kept under the "
            f"'{mode}' gate ({MODE_LABELS[mode]})"
        )

    # --- Statistical Output & Visualization ---
    colors = {
        FIXED_LABEL: "gray",
        "Single Pixel": "blue",
        "Max 3x3": "green",
        "Uphill Max": "firebrick",
    }
    linestyles = {"hydro": "-", "full": "--"}

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    metric_rows = []

    for mode in MODES:
        print(f"\n--- Final Spatial Results ({MODE_LABELS[mode]}) ---")
        print(
            f"{'variant':>15} | {'AUC':>6} | {'J':>6} | {'MCC':>6} | {'TPR':>6} | "
            f"{'FPR':>6} | {'n':>6}"
        )
        for variant, data in results[mode].items():
            pos = np.array(data["pos"])
            neg = np.array(data["neg"])

            if len(pos) == 0:
                print(f"{variant}: Insufficient data.")
                continue

            auc = autils.auc_score(pos, neg)
            event_flags, control_flags = pos <= 1.0, neg <= 1.0
            tpr, fpr = vm.tpr_fpr(event_flags, control_flags)
            j = vm.youden_j(event_flags, control_flags)
            m = vm.mcc(event_flags, control_flags)
            print(
                f"{variant:>15} | {auc:6.3f} | {j:6.3f} | {m:6.3f} | {tpr:6.3f} | "
                f"{fpr:6.3f} | {len(pos):6d}"
            )
            metric_rows.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "auc": auc,
                    "youden_j": j,
                    "mcc": m,
                    "tpr_at_fos1": tpr,
                    "fpr_at_fos1": fpr,
                    "n": len(pos),
                }
            )

            thr = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
            tpr_curve = [(pos <= t).mean() for t in thr]
            fpr_curve = [(neg <= t).mean() for t in thr]

            ax_roc.plot(
                fpr_curve,
                tpr_curve,
                label=f"{variant} ({mode}, AUC = {auc:.3f})",
                color=colors[variant],
                linestyle=linestyles[mode],
                linewidth=2,
            )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    ax_roc.set_title(
        "ROC/AUC Comparison of Spatial Integration Variants\n",
        fontsize=13,
    )
    ax_roc.set_xlabel("False Positive Rate (Control Days Triggered)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (Disaster Days Triggered)", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    output_path = f"{OUTPUT_DIR}/spatial_roc_comparison.png"
    fig_roc.tight_layout()
    fig_roc.savefig(output_path, dpi=150)
    plt.close(fig_roc)
    print(f"\nSaved combined ROC plot to: {output_path}")

    if not metric_rows:
        return

    metrics_df = pd.DataFrame(metric_rows)
    metrics_csv = f"{OUTPUT_DIR}/spatial_variant_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics table to: {metrics_csv}")

    # --- Bar charts: AUC, Youden's J, MCC per variant -- hydro (left) vs full (right) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, mode in zip(axes, MODES):
        sub = metrics_df[metrics_df["mode"] == mode].reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{MODE_LABELS[mode]} (no data)")
            continue
        x = np.arange(len(sub))
        width = 0.25
        ax.bar(x - width, sub["auc"], width, label="AUC", color="steelblue")
        ax.bar(x, sub["youden_j"], width, label="Youden's J", color="purple")
        ax.bar(x + width, sub["mcc"], width, label="MCC", color="darkorange")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["variant"], rotation=15)
        ax.set_title(MODE_LABELS[mode])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend()
    fig.suptitle("AUC vs threshold metrics (FoS = 1) per spatial variant", y=1.02)
    fig.tight_layout()
    metrics_png = f"{OUTPUT_DIR}/spatial_variant_metrics.png"
    fig.savefig(metrics_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics bar chart to: {metrics_png}")


if __name__ == "__main__":
    main()
