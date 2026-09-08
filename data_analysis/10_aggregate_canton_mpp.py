"""
10_aggregate_canton_mpp.py - Aggregate canton-level historical m_pp statistics.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import constants as const
from core import data_loader as dl
from core import physics

# ============================================================= CONFIG =====

OUTDIR = "output/data_analysis/10_canton_mpp"
YEARS = range(1991, 2025)

N_SAMPLE_POINTS = 25
MIN_DISTANCE_M = 5000.0
MAX_SNAP_M = 2000.0
RANDOM_SEED = 42

STATISTICS_FILE = f"{OUTDIR}/canton_mpp_statistics.csv"
YEARLY_FILE = f"{OUTDIR}/canton_mpp_yearly.csv"
MONTHLY_FILE = f"{OUTDIR}/canton_mpp_monthly.csv"
CLIMATOLOGY_FILE = f"{OUTDIR}/canton_mpp_climatology.csv"
POINTS_FILE = f"{OUTDIR}/canton_sampling_points.csv"

MEAN_MAP_FILE = f"{OUTDIR}/mean_mpp_by_canton.png"
P95_MAP_FILE = f"{OUTDIR}/p95_mpp_by_canton.png"
ACTIVE_MAP_FILE = f"{OUTDIR}/active_days_by_canton.png"

os.makedirs(OUTDIR, exist_ok=True)


# =========================================================== SAMPLING =====


def sample_canton_points(
    canton_raster,
    transform,
    canton_id,
    n_points=N_SAMPLE_POINTS,
    min_distance_m=MIN_DISTANCE_M,
    seed=RANDOM_SEED,
):
    """Select spatially distributed sample points inside one canton."""
    rows, cols = np.where(canton_raster == canton_id)

    if len(rows) == 0:
        return []

    rng = np.random.default_rng(seed + int(canton_id))
    candidate_idx = np.arange(len(rows))
    rng.shuffle(candidate_idx)

    selected = []

    for idx in candidate_idx:
        row, col = rows[idx], cols[idx]
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        x, y = float(x), float(y)

        if min_distance_m is not None and selected:
            distances = [np.hypot(x - sx, y - sy) for sx, sy in selected]
            if min(distances) < min_distance_m:
                continue

        selected.append((x, y))

        if len(selected) >= n_points:
            break

    # Fill remaining points if the spacing constraint was too restrictive
    if len(selected) < n_points:
        existing = set(selected)

        for idx in candidate_idx:
            row, col = rows[idx], cols[idx]
            x, y = rasterio.transform.xy(transform, row, col, offset="center")
            point = (float(x), float(y))

            if point not in existing:
                selected.append(point)
                existing.add(point)

            if len(selected) >= n_points:
                break

    return selected


# ========================================================= POINT MODEL =====


def calculate_mpp_at_point(easting, northing):
    """Calculate historical daily saturation and m_pp at one point."""
    drought_region_id = dl.assign_region(easting, northing, max_snap_m=MAX_SNAP_M)

    if drought_region_id is None:
        return None

    drainage, et, et_amp = dl.load_calibration_params(drought_region_id)

    if drainage is None or et is None:
        print(
            f"      drought region {drought_region_id}: missing calibration -> using defaults"
        )
        drainage, et, et_amp = 0.1, 2.0, 0.0

    if et_amp is None:
        et_amp = 0.0

    rainfall = dl.load_rainfall(easting, northing, YEARS)

    if rainfall is None or rainfall.empty:
        return None

    rainfall = rainfall.sort_index()

    with redirect_stdout(io.StringIO()):
        saturation = physics.calculate_daily_saturation(
            precip_mm_day=rainfall.to_numpy(dtype=float),
            n=const.N,
            n_perp=const.H_PERP,
            m0=const.M0,
            s_pp_onset=const.S_PP_ONSET_DEFAULT,
            drainage_rate=drainage,
            et_rate=et,
            day_of_year=rainfall.index.dayofyear.to_numpy(),
            et_amplitude=et_amp,
        )

    m_pp = physics.pore_pressure_ratio(saturation, const.S_PP_ONSET_DEFAULT)

    result = pd.DataFrame(
        {
            "saturation_ratio": saturation,
            "m_pp": m_pp,
            "drought_region_id": drought_region_id,
        },
        index=rainfall.index,
    )

    result.index.name = "measured_at"
    return result


# ======================================================== CANTON MODEL =====


def compute_canton_daily(canton_id, canton_name, canton_raster, transform):
    """Calculate daily spatially averaged saturation and m_pp for one canton."""
    points = sample_canton_points(canton_raster, transform, canton_id)

    print(f"\nCanton {canton_id} ({canton_name}): {len(points)} sample points")

    saturation_series = []
    mpp_series = []
    point_records = []

    for point_id, (x, y) in enumerate(points, start=1):
        point_data = calculate_mpp_at_point(x, y)

        if point_data is None:
            print(f"    point {point_id:02d}: ({x:.0f}, {y:.0f}) -> no usable data")

            point_records.append(
                {
                    "canton_id": canton_id,
                    "canton_name": canton_name,
                    "point_id": point_id,
                    "easting": x,
                    "northing": y,
                    "drought_region_id": np.nan,
                    "valid": False,
                }
            )
            continue

        drought_region_id = int(point_data["drought_region_id"].iloc[0])
        mean_s = point_data["saturation_ratio"].mean()
        mean_mpp = point_data["m_pp"].mean()

        print(
            f"    point {point_id:02d}: ({x:.0f}, {y:.0f}) -> region {drought_region_id}, mean S={mean_s:.3f}, mean m_pp={mean_mpp:.3f}"
        )

        saturation_series.append(
            point_data["saturation_ratio"].rename(f"point_{point_id}")
        )
        mpp_series.append(point_data["m_pp"].rename(f"point_{point_id}"))

        point_records.append(
            {
                "canton_id": canton_id,
                "canton_name": canton_name,
                "point_id": point_id,
                "easting": x,
                "northing": y,
                "drought_region_id": drought_region_id,
                "valid": True,
            }
        )

    if not mpp_series:
        print("    No valid sampling points.")
        return None, point_records

    saturation_df = pd.concat(saturation_series, axis=1)
    mpp_df = pd.concat(mpp_series, axis=1)

    daily = pd.DataFrame(
        {
            "saturation_ratio": saturation_df.mean(axis=1),
            "m_pp": mpp_df.mean(axis=1),
            "n_points": mpp_df.notna().sum(axis=1),
        }
    )

    daily.index.name = "measured_at"
    return daily, point_records


# ========================================================== STATISTICS =====


def canton_statistics(canton_id, canton_name, daily):
    """Calculate historical canton-level m_pp statistics."""
    mpp = daily["m_pp"]
    saturation = daily["saturation_ratio"]

    return {
        "canton_id": canton_id,
        "canton_name": canton_name,
        "n_days": len(daily),
        "mean_saturation": saturation.mean(),
        "median_saturation": saturation.median(),
        "mean_m_pp": mpp.mean(),
        "median_m_pp": mpp.median(),
        "p75_m_pp": mpp.quantile(0.75),
        "p90_m_pp": mpp.quantile(0.90),
        "p95_m_pp": mpp.quantile(0.95),
        "p99_m_pp": mpp.quantile(0.99),
        "max_m_pp": mpp.max(),
        "fraction_active_days": (mpp > 0.0).mean(),
        "fraction_mpp_ge_025": (mpp >= 0.25).mean(),
        "fraction_mpp_ge_050": (mpp >= 0.50).mean(),
        "fraction_mpp_ge_075": (mpp >= 0.75).mean(),
    }


# ========================================================== MAP UTILS =====


def build_metric_map(canton_id_raster, values_by_id):
    """Paint each canton with its corresponding metric value."""
    max_id = int(canton_id_raster.max())
    lookup = np.full(max_id + 1, np.nan, dtype=np.float32)

    for cid, value in values_by_id.items():
        if 0 <= cid <= max_id:
            lookup[cid] = value

    idx = np.clip(canton_id_raster, 0, max_id)
    metric_map = lookup[idx]
    metric_map[canton_id_raster <= 0] = np.nan

    return metric_map


def detect_borders(id_raster, thickness_px=4):
    """Derive canton borders from the canton-ID raster."""
    border = np.zeros_like(id_raster, dtype=bool)

    diff_v = id_raster[:-1, :] != id_raster[1:, :]
    border[:-1, :] |= diff_v
    border[1:, :] |= diff_v

    diff_h = id_raster[:, :-1] != id_raster[:, 1:]
    border[:, :-1] |= diff_h
    border[:, 1:] |= diff_h

    return binary_dilation(border, iterations=thickness_px)


def crop_to_data_bbox(id_raster, transform):
    """Crop raster to Switzerland's nonzero footprint."""
    from rasterio.windows import Window, bounds as window_bounds

    rows, cols = np.nonzero(id_raster > 0)

    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    cropped = id_raster[r0:r1, c0:c1]
    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)

    left, bottom, right, top = window_bounds(window, transform)
    return cropped, (left, right, bottom, top)


# ================================================================ MAPS =====


def plot_canton_mpp_map(statistics_df, canton_raster, transform, metric, output_file):
    """Plot a canton-level historical m_pp choropleth."""
    if metric not in statistics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in statistics DataFrame.")

    values_by_id = statistics_df.set_index("canton_id")[metric].to_dict()

    canton_crop, (left, right, bottom, top) = crop_to_data_bbox(
        canton_raster, transform
    )
    border_mask = detect_borders(canton_crop, thickness_px=4)
    metric_map = build_metric_map(canton_crop, values_by_id)

    labels = {
        "mean_m_pp": "Historical mean pore-pressure activation $m_{pp}$",
        "median_m_pp": "Historical median pore-pressure activation $m_{pp}$",
        "p90_m_pp": "Historical 90th percentile $m_{pp}$",
        "p95_m_pp": "Historical 95th percentile $m_{pp}$",
        "p99_m_pp": "Historical 99th percentile $m_{pp}$",
        "fraction_active_days": "Fraction of days with active pore pressure",
    }

    titles = {
        "mean_m_pp": "Historical Mean Pore-Pressure Activation by Canton",
        "median_m_pp": "Historical Median Pore-Pressure Activation by Canton",
        "p90_m_pp": "Historical P90 Pore-Pressure Activation by Canton",
        "p95_m_pp": "Historical P95 Pore-Pressure Activation by Canton",
        "p99_m_pp": "Historical P99 Pore-Pressure Activation by Canton",
        "fraction_active_days": "Historical Pore-Pressure Activation Frequency by Canton",
    }

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        metric_map,
        cmap="YlGnBu",
        interpolation="nearest",
        extent=[left, right, bottom, top],
        vmin=0.0,
        vmax=1.0,
    )

    border_display = np.ma.masked_where(
        ~border_mask, np.ones_like(border_mask, dtype=float)
    )

    ax.imshow(
        border_display,
        cmap=ListedColormap(["black"]),
        extent=[left, right, bottom, top],
        interpolation="nearest",
        zorder=3,
        alpha=0.8,
    )

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(labels.get(metric, metric))

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        titles.get(metric, f"{metric} by Canton"), fontsize=13, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"  -> {output_file}")


# ============================================================ AGGREGATE =====


def aggregate_canton_mpp():
    """Calculate historical m_pp statistics for all cantons."""
    canton_raster, transform, bounds = dl.load_canton_raster()
    canton_names = dl.load_canton_names()

    canton_ids = sorted(int(cid) for cid in np.unique(canton_raster) if cid > 0)

    print("=" * 65)
    print(" HISTORICAL CANTON m_pp AGGREGATION")
    print("=" * 65)
    print(f"Cantons: {len(canton_ids)}")
    print(f"Rainfall period: {min(YEARS)}-{max(YEARS)}")
    print(f"Sampling points per canton: {N_SAMPLE_POINTS}")
    print(f"Pore-pressure onset: {const.S_PP_ONSET_DEFAULT:.3f}")

    all_daily = []
    statistics = []
    sampling_records = []

    for canton_id in canton_ids:
        canton_name = canton_names.get(canton_id, f"Canton {canton_id}")

        daily, points = compute_canton_daily(
            canton_id, canton_name, canton_raster, transform
        )
        sampling_records.extend(points)

        if daily is None:
            continue

        stats = canton_statistics(canton_id, canton_name, daily)
        statistics.append(stats)

        print(
            f"    RESULT: mean={stats['mean_m_pp']:.3f}, P95={stats['p95_m_pp']:.3f}, P99={stats['p99_m_pp']:.3f}, active={100 * stats['fraction_active_days']:.1f}%"
        )

        daily = daily.reset_index()
        daily["canton_id"] = canton_id
        daily["canton_name"] = canton_name
        all_daily.append(daily)

    if not all_daily:
        raise RuntimeError("No canton produced valid historical m_pp data.")

    # Sampling points
    sampling_df = pd.DataFrame(sampling_records)
    sampling_df.to_csv(POINTS_FILE, index=False)

    # Daily data
    daily_all = pd.concat(all_daily, ignore_index=True)
    daily_all["measured_at"] = pd.to_datetime(daily_all["measured_at"])

    # Overall statistics
    statistics_df = (
        pd.DataFrame(statistics).sort_values("canton_id").reset_index(drop=True)
    )
    statistics_df.to_csv(STATISTICS_FILE, index=False)

    # Yearly statistics
    daily_all["year"] = daily_all["measured_at"].dt.year

    yearly = (
        daily_all.groupby(["canton_id", "canton_name", "year"])
        .agg(
            mean_saturation=("saturation_ratio", "mean"),
            mean_m_pp=("m_pp", "mean"),
            median_m_pp=("m_pp", "median"),
            p95_m_pp=("m_pp", lambda x: x.quantile(0.95)),
            max_m_pp=("m_pp", "max"),
            active_fraction=("m_pp", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    yearly.to_csv(YEARLY_FILE, index=False)

    # Monthly statistics
    daily_all["year_month"] = daily_all["measured_at"].dt.to_period("M").astype(str)

    monthly = (
        daily_all.groupby(["canton_id", "canton_name", "year_month"])
        .agg(
            mean_saturation=("saturation_ratio", "mean"),
            mean_m_pp=("m_pp", "mean"),
            median_m_pp=("m_pp", "median"),
            p95_m_pp=("m_pp", lambda x: x.quantile(0.95)),
            max_m_pp=("m_pp", "max"),
            active_fraction=("m_pp", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    monthly.to_csv(MONTHLY_FILE, index=False)

    # Monthly climatology
    daily_all["month"] = daily_all["measured_at"].dt.month

    climatology = (
        daily_all.groupby(["canton_id", "canton_name", "month"])
        .agg(
            mean_saturation=("saturation_ratio", "mean"),
            mean_m_pp=("m_pp", "mean"),
            median_m_pp=("m_pp", "median"),
            p90_m_pp=("m_pp", lambda x: x.quantile(0.90)),
            p95_m_pp=("m_pp", lambda x: x.quantile(0.95)),
            active_fraction=("m_pp", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    climatology.to_csv(CLIMATOLOGY_FILE, index=False)

    # Maps
    print("\nGenerating canton maps...")

    plot_canton_mpp_map(
        statistics_df, canton_raster, transform, "mean_m_pp", MEAN_MAP_FILE
    )
    plot_canton_mpp_map(
        statistics_df, canton_raster, transform, "p95_m_pp", P95_MAP_FILE
    )
    plot_canton_mpp_map(
        statistics_df, canton_raster, transform, "fraction_active_days", ACTIVE_MAP_FILE
    )

    # Summary
    print("\n" + "=" * 65)
    print(" COMPLETE")
    print("=" * 65)

    display_cols = [
        "canton_name",
        "mean_m_pp",
        "p90_m_pp",
        "p95_m_pp",
        "p99_m_pp",
        "fraction_active_days",
    ]

    print("\nHistorical canton m_pp statistics:\n")
    print(
        statistics_df[display_cols]
        .sort_values("mean_m_pp", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    aggregate_canton_mpp()
