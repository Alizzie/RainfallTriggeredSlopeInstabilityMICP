"""
root_cohesion.py - Phase 3: effect of root cohesion, decoupled from MICP.

Output (OUTPUT_DIR = output/spatial/06_root_cohesion):
    root_cohesion_variant_table.csv  forest class name -> kPa, per tier (the
                                      real numbers behind the "low/mid/high" labels)
    worst_case_rescue.csv            national unstable share, per (H_PERP, cohesion variant)
    worst_case_rescue_vs_h_perp.png  rescued share vs H_PERP, one line per cohesion variant
    temporal_validation.csv          AUC/J/MCC per (H_PERP, slope variant, cohesion variant)
    temporal_validation_<variant>.png  one 1x3 (AUC | J | MCC) figure per slope variant
"""

import sys
import os
import io
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import physics
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

OUTPUT_DIR = "output/spatial/06_root_cohesion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Failure-depth values to sweep alongside cohesion. 0.3-1.1m matches the range
# sensitivity_analysis.py already tests (StorMe field measurements cite
# 0.7-1.1m at real scars); const.H_PERP (0.6m, the model default) is included
# so every result here has a baseline to compare against.
H_PERP_GRID = sorted({0.3, 0.6, const.H_PERP, 1.0, 1.1, 1.5, 2})

# Forest raster class -> added cohesion [kPa]. Class codes and German names
# match SwissTLM3D; "low"/"mid"/"high" scale the same four classes to bracket
# the model's own ceiling (see module docstring). Plot labels stay short
# ("low"/"mid"/"high") -- the actual per-class kPa values are always
# printed/saved via print_root_cohesion_table(), never hidden in the label.
FOREST_CLASS_NAMES = {
    6: "Gebueschwald",
    12: "Wald",
    13: "Wald offen",
    14: "Gehoelzflaeche",
}

ROOT_COHESION_VARIANTS = {
    "low (1-3 kPa)": {6: 1.0, 13: 1.5, 14: 2.0, 12: 3.0},
    "mid (6-15 kPa)": {6: 6.0, 13: 8.0, 14: 10.0, 12: 15.0},
    "high (10-22 kPa)": {6: 10.0, 13: 14.0, 14: 18.0, 12: 22.0},
}


def print_root_cohesion_table():
    """Human-readable reference table: forest class name -> cohesion [kPa],
    per variant. Saved once so plot legends can stay short ("low"/"mid"/
    "high") without hiding what those labels actually mean.
    """
    rows = [
        {
            "forest_class_code": cls,
            "forest_class_name": FOREST_CLASS_NAMES.get(cls, f"class {cls}"),
            "tier": tier,
            "cohesion_kpa": kpa,
        }
        for tier, mapping in ROOT_COHESION_VARIANTS.items()
        for cls, kpa in mapping.items()
    ]
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUTPUT_DIR}/root_cohesion_variant_table.csv", index=False)
    print("\nRoot cohesion by forest class [kPa]:")
    print(
        table.pivot(index="forest_class_name", columns="tier", values="cohesion_kpa")
        .reindex(columns=list(ROOT_COHESION_VARIANTS.keys()))
        .to_string()
    )
    return table


def root_cohesion_from_class(forest_class_array, variant):
    """Map a forest-class raster to added cohesion [kPa], 0 outside any listed class."""
    added = np.zeros(forest_class_array.shape, dtype=np.float32)
    for forest_class, kpa in variant.items():
        added[forest_class_array == forest_class] = kpa
    return added


# ============================================================ A) national map


def worst_case_rescue():
    """Unstable share at m_pp=1: bare soil vs bare+roots, per (H_PERP, root-
    cohesion variant). H_PERP sets the model's own cohesion ceiling (see
    module docstring), so it's the direct lever for testing whether that
    ceiling -- not root cohesion itself -- is what's limiting the rescue.

    bare_unstable is computed ONCE, outside the H_PERP loop: at c'=0, H
    cancels out of the FoS formula exactly (H appears linearly in both the
    normal-stress and shear-stress terms), so which pixels are unstable on
    bare soil does not depend on failure depth at all -- confirmed numerically
    (bare_unstable's count is bit-identical across every H_PERP tested).
    """
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta_deg = src.read(1).astype(np.float32)
        slope_nodata = src.nodata

    forest_class, forest_transform, forest_nodata = dl.load_forest_raster()
    if forest_class.shape != beta_deg.shape:
        raise SystemExit(
            f"forest raster shape {forest_class.shape} != slope raster shape "
            f"{beta_deg.shape} -- they must share one grid."
        )

    valid = np.isfinite(beta_deg)
    if slope_nodata is not None:
        valid &= beta_deg != slope_nodata
    valid &= (beta_deg >= auct.BETA_MIN) & (beta_deg <= auct.BETA_MAX)

    beta_rad = np.radians(beta_deg[valid])
    forest_valid = forest_class[valid]
    n_valid = int(valid.sum())

    # H-independent at c'=0 (see docstring) -- compute once.
    fos0_worst_case_any_h = physics.compute_fos(
        m_array=1.0,
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=1.0
        / np.cos(beta_rad),  # H_PERP cancels; any nonzero value gives the same result
        beta_rad=beta_rad,
        phi_rad=const.phi,
    )
    bare_unstable = fos0_worst_case_any_h < 1.0
    n_bare_unstable = int(bare_unstable.sum())

    is_forest = np.isin(forest_valid, list(FOREST_CLASS_NAMES.keys()))
    bare_unstable_non_forest = bare_unstable & ~is_forest
    non_forest_floor = float(bare_unstable_non_forest.sum() / max(n_bare_unstable, 1))
    print(
        f"\nOf bare-unstable pixels, {non_forest_floor:.1%} are non-forest "
        f"(bare rock, scree, alpine terrain above treeline, farmland, etc.). "
        f"Root cohesion only applies where forest exists -- this share is the "
        f"floor NO root-cohesion variant, at any H_PERP, can ever rescue; only "
        f"a treatment that doesn't require forest (MICP) can reach it."
    )

    rows = []
    for h_perp in H_PERP_GRID:
        h_v = h_perp / np.cos(beta_rad)
        fos0_worst_case = physics.compute_fos(
            m_array=1.0,  # design saturation: fully saturated, the worst case
            c=0.0,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=h_v,
            beta_rad=beta_rad,
            phi_rad=const.phi,
        )
        tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)

        def fos_at(cohesion_kpa):
            return fos0_worst_case + cohesion_kpa / tau

        for name, mapping in ROOT_COHESION_VARIANTS.items():
            added = root_cohesion_from_class(forest_valid, mapping)
            unstable = fos_at(const.C + added) < 1.0
            rescued = bare_unstable & ~unstable
            rows.append(
                {
                    "h_perp_m": h_perp,
                    "variant": name,
                    "unstable_share": float(unstable.sum() / n_valid),
                    "rescued_of_bare_unstable": float(
                        rescued.sum() / max(n_bare_unstable, 1)
                    ),
                }
            )
    # bare soil row: identical at every H_PERP (see docstring), added once per H
    # for a complete CSV without recomputing anything.
    for h_perp in H_PERP_GRID:
        rows.append(
            {
                "h_perp_m": h_perp,
                "variant": "bare soil only",
                "unstable_share": n_bare_unstable / n_valid,
                "rescued_of_bare_unstable": 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUTPUT_DIR}/worst_case_rescue.csv", index=False)
    print("\n--- A) National worst case (m_pp = 1), per H_PERP ---")
    print(
        f"valid pixels: {n_valid:,}   bare-unstable: {n_bare_unstable:,} ({n_bare_unstable/n_valid:.1%})"
    )
    print(result.sort_values(["h_perp_m", "variant"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in ROOT_COHESION_VARIANTS:
        sub = result[result.variant == name].sort_values("h_perp_m")
        ax.plot(
            sub.h_perp_m, sub.rescued_of_bare_unstable * 100, "o-", lw=2, label=name
        )
    ax.axhline(
        (1 - non_forest_floor) * 100,
        color="gray",
        ls="--",
        alpha=0.6,
        label="max possible (forested share of bare-unstable)",
    )
    ax.axvline(
        const.H_PERP,
        color="black",
        ls=":",
        alpha=0.6,
        label=f"model default ({const.H_PERP} m)",
    )
    ax.set_xlabel("H_PERP (m)")
    ax.set_ylabel("% of bare-unstable pixels rescued")
    ax.set_title("Root-cohesion rescue vs assumed failure depth (m_pp = 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/worst_case_rescue_vs_h_perp.png", dpi=150)
    plt.close(fig)

    return result


# ============================================================ B) temporal validation
#
# Tests {H_PERP} x {slope-sampling variant} x {root-cohesion variant} together.
# Each slope variant keeps its OWN independently-valid subset (an event is
# excluded from a variant only if THAT variant's own angle falls outside
# [15, 45] deg) -- this is not a strict common-subset ROC comparison between
# slope variants (see 04_compare_spatial_variants.py for that); the goal here
# is "does adding root cohesion help THIS slope variant, at THIS failure
# depth", not "which slope variant wins" (already answered in Phase 1). Note
# that subset does NOT depend on H_PERP (the angle-validity filter is angle-
# only), so n_events/n_controls are identical across H_PERP for a given slope
# variant -- only the physics changes, not the population.


UPHILL_RADIUS_M = None  # resolved in main() once, see resolve_uphill_radius()


def resolve_uphill_radius(default=60):
    """Same as 04_compare_spatial_variants.py: pick the radius from
    03_sweep_uphill_radius.py's hydro-only AUC result."""
    csv_path = "output/spatial/03_uphill_sweep/auc_vs_radius.csv"
    if not os.path.exists(csv_path):
        print(f"[warn] {csv_path} not found -- using default radius={default} m.")
        return default
    df = pd.read_csv(csv_path)
    return int(df.loc[df["auc_hydro"].idxmax(), "radius_m"])


def get_max_3x3_slope(slope_raster, row, col):
    from rasterio.windows import Window

    data = slope_raster.read(
        1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan
    ).astype(float)
    if slope_raster.nodata is not None:
        data[data == slope_raster.nodata] = np.nan
    in_band = (data >= auct.BETA_MIN) & (data <= auct.BETA_MAX) & np.isfinite(data)
    candidates = data[in_band]
    return float(candidates.max()) if candidates.size > 0 else np.nan


def get_uphill_max_slope(dem_raster, slope_raster, row, col, x, y, radius_m):
    from rasterio.windows import Window

    pixel_size = dem_raster.res[0]
    r_pix = int(round(radius_m / pixel_size))
    win_size = 2 * r_pix + 1
    window = Window(col - r_pix, row - r_pix, win_size, win_size)

    dem_data = dem_raster.read(
        1, window=window, boundless=True, fill_value=np.nan
    ).astype(float)
    slope_data = slope_raster.read(
        1, window=window, boundless=True, fill_value=np.nan
    ).astype(float)
    if dem_raster.nodata is not None:
        dem_data[dem_data == dem_raster.nodata] = np.nan
    if slope_raster.nodata is not None:
        slope_data[slope_data == slope_raster.nodata] = np.nan

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


def max_pore_pressure_ratio(x, y, date):
    """Max pore-pressure ratio in the +/- window around date, or NaN if no rainfall data.
    Same closed form as soil_cohesion.py -- raw (x, y), NOT converted to LV95:
    get_region_params / load_rainfall take the inventory's native coordinates
    directly (see soil_cohesion.py, which never converts either). Conversion
    is only needed below, for indexing into the LV95 rasters."""
    _, drainage, et, et_amp = dl.get_region_params(x, y, auct.CALIB)
    if drainage is None:
        return np.nan

    start = date - pd.Timedelta(days=auct.SPINUP_DAYS)
    end = date + pd.Timedelta(days=auct.WINDOW_DAYS + 5)
    rain = dl.load_rainfall(x, y, sorted({start.year, end.year}))
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


def collect_case_control_points():
    """(pore_pressure_ratio, forest_class, {variant: angle}) for every WSL
    event and its screened controls. Angle depends only on (x, y), so every
    control at a location shares its event's angles and forest class.
    """
    inventory = dl.load_wsl_usable_inventory()
    forest_class_full, _, _ = dl.load_forest_raster()

    with rasterio.open(auct.SLOPE_TIF) as slope_src, rasterio.open(
        auct.DEM_TIF
    ) as dem_src:
        if forest_class_full.shape != slope_src.shape:
            raise SystemExit("forest raster and slope raster must share one grid.")

        event_points, control_points = [], []
        for i, event in inventory.iterrows():
            x, y = event["x"], event["y"]
            x_lv95, y_lv95 = dl.to_lv95(x, y)
            row, col = slope_src.index(x_lv95, y_lv95)

            forest_class = (
                int(forest_class_full[row, col])
                if 0 <= row < forest_class_full.shape[0]
                and 0 <= col < forest_class_full.shape[1]
                else 0  # off-raster treated as non-forest
            )
            angles = {
                f"Fixed {const.BETA_DEG}\u00b0": float(const.BETA_DEG),
                "Max 3x3": get_max_3x3_slope(slope_src, row, col),
                "Uphill Max": get_uphill_max_slope(
                    dem_src, slope_src, row, col, x_lv95, y_lv95, UPHILL_RADIUS_M
                ),
            }

            event_pp = max_pore_pressure_ratio(x, y, event["date"])
            controls = vm.find_control_date(
                x, y, event["date"], inventory, n=auct.CONTROLS_PER_EVENT
            )
            control_pps = [max_pore_pressure_ratio(x, y, c) for c in controls]

            if np.isfinite(event_pp):
                event_points.append((event_pp, forest_class, angles))
            for m in control_pps:
                if np.isfinite(m):
                    control_points.append((m, forest_class, angles))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(inventory)} processed")

    return event_points, control_points


def score_variant_pair(
    event_points, control_points, slope_variant, cohesion_kpa_lookup, h_perp
):
    """AUC/J/MCC for one (slope variant, cohesion variant, H_PERP) triple.

    Drops points whose angle is invalid (outside [15, 45] deg or NaN) for
    THIS slope variant -- see the section docstring for why this is an
    independent-per-variant subset, not a shared common subset.
    """

    def usable(points):
        pp, forest, angle = [], [], []
        for m, forest_class, angles in points:
            a = angles[slope_variant]
            if autils.classify_slope(a) == "invalid":
                continue
            pp.append(m)
            forest.append(forest_class)
            angle.append(a)
        return np.array(pp), np.array(forest), np.array(angle)

    event_pp, event_forest, event_beta = usable(event_points)
    control_pp, control_forest, control_beta = usable(control_points)
    if len(event_pp) == 0 or len(control_pp) == 0:
        return None

    def fos(pore_pressure, forest_class, beta_deg):
        beta_rad = np.radians(beta_deg)
        h_v = h_perp / np.cos(beta_rad)
        tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
        added = root_cohesion_from_class(forest_class, cohesion_kpa_lookup)
        fos0 = physics.compute_fos(
            m_array=pore_pressure,
            c=0.0,
            gamma=const.GAMMA,
            gamma_w=const.GAMMA_W,
            h_v=h_v,
            beta_rad=beta_rad,
            phi_rad=const.phi,
        )
        return fos0 + (const.C + added) / tau

    fos_events = fos(event_pp, event_forest, event_beta)
    fos_controls = fos(control_pp, control_forest, control_beta)
    event_flags, control_flags = fos_events <= 1.0, fos_controls <= 1.0

    return {
        "auc": autils.auc_score(fos_events, fos_controls),
        "youden_j": vm.youden_j(event_flags, control_flags),
        "mcc": vm.mcc(event_flags, control_flags),
        "n_events": len(event_pp),
        "n_controls": len(control_pp),
    }


def temporal_validation():
    global UPHILL_RADIUS_M
    UPHILL_RADIUS_M = resolve_uphill_radius()
    print(f"Using uphill radius {UPHILL_RADIUS_M} m")

    slope_variants = [f"Fixed {const.BETA_DEG}\u00b0", "Max 3x3", "Uphill Max"]
    cohesion_variants = {"bare soil only": {}, **ROOT_COHESION_VARIANTS}
    cohesion_order = list(cohesion_variants.keys())

    print(
        "\n--- B) Temporal validation: H_PERP x slope variant x root-cohesion variant ---"
    )
    event_points, control_points = collect_case_control_points()

    rows = []
    for h_perp in H_PERP_GRID:
        for slope_variant in slope_variants:
            for cohesion_name, mapping in cohesion_variants.items():
                scored = score_variant_pair(
                    event_points, control_points, slope_variant, mapping, h_perp
                )
                if scored is None:
                    continue
                rows.append(
                    dict(
                        h_perp_m=h_perp,
                        slope_variant=slope_variant,
                        cohesion_variant=cohesion_name,
                        **scored,
                    )
                )

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUTPUT_DIR}/temporal_validation.csv", index=False)
    print(result.to_string(index=False))

    # One 1x3 (AUC | Youden's J | MCC) figure PER slope variant, one line per H_PERP.
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(H_PERP_GRID)))
    metrics = [("auc", "AUC"), ("youden_j", "Youden's J"), ("mcc", "MCC")]

    for slope_variant in slope_variants:
        sub_variant = result[result.slope_variant == slope_variant]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        for ax, (col_name, title) in zip(axes, metrics):
            for h_perp, color in zip(H_PERP_GRID, colors):
                sub = (
                    sub_variant[sub_variant.h_perp_m == h_perp]
                    .set_index("cohesion_variant")
                    .reindex(cohesion_order)
                )
                label = f"H={h_perp} m" + (
                    " (default)" if h_perp == const.H_PERP else ""
                )
                ax.plot(cohesion_order, sub[col_name], "o-", color=color, label=label)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=25)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        axes[0].set_ylabel("score")
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle(
            f"{slope_variant}: root cohesion vs temporal skill, across H_PERP", y=1.03
        )
        fig.tight_layout()
        slug = slope_variant.lower().replace(" ", "_").replace("\u00b0", "deg")
        fig.savefig(
            f"{OUTPUT_DIR}/temporal_validation_{slug}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    return result


def main():
    print_root_cohesion_table()
    worst_case_rescue()
    temporal_validation()
    print(
        f"\n-> {OUTPUT_DIR}/ (root_cohesion_variant_table.csv, worst_case_rescue.csv/.png, "
        f"temporal_validation.csv + one .png per slope variant)"
    )


if __name__ == "__main__":
    main()
