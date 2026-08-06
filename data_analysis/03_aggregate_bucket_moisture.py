"""Aggregate bucket-model soil moisture (historical rainfall) by year and month, mirroring the field-capacity outputs."""

import io
import os
import sys
from contextlib import redirect_stdout

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import constants as const
from core import data_loader as dl
from core import physics
from core import region_map as rm

OUTDIR = "output/data_analysis/03_aggregate_bucket_moisture"
YEARLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_yearly.csv"
MONTHLY_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_monthly.csv"
BY_REGION_OUTPUT_FILE = f"{OUTDIR}/mean_moisture_by_region.csv"

# Years of RhiresD rainfall used to drive the bucket model.
YEARS = range(1991, 2025)
os.makedirs(OUTDIR, exist_ok=True)


def region_ids_from_calibration():
    """
    Region IDs to simulate: those present in BOTH the calibration table and the
    region polygons. Falls back to all polygons if calibration is unavailable.
    """
    geometries = dl.load_region_geometries()
    if not os.path.exists(dl.PATH_CALIB):
        return sorted(geometries)
    calib = pd.read_csv(dl.PATH_CALIB)
    ids = [int(r) for r in calib["region_id"].unique() if int(r) in geometries]
    return sorted(ids)


def _saturation_at_point(easting, northing, drainage, et, et_amp):
    """Run the bucket model on rainfall at one (E, N) point. None if no data."""
    rainfall = dl.load_rainfall(easting, northing, YEARS)
    if rainfall is None or rainfall.empty:
        return None

    # Bucket model is chatty; swallow its per-call banner to keep the log clean.
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

    return pd.Series(saturation, index=rainfall.index, name="saturation_ratio")


def compute_region_saturation(region_id, geometry):
    """
    Daily saturation ratio S(t) for one region from the bucket model.
    """
    drainage, et, et_amp = dl.load_calibration_params(region_id)
    if drainage is None or et is None:
        drainage, et, et_amp = 0.1, 2.0, 0.0  # physics defaults

    weights = rm.part_area_weights(geometry)
    series_list = []
    for (easting, northing), weight in weights:
        part_series = _saturation_at_point(easting, northing, drainage, et, et_amp)
        if part_series is not None:
            series_list.append((part_series, weight))

    if not series_list:
        return None
    if len(series_list) == 1:
        return series_list[0][0]

    # Re-normalise weights over the parts that actually returned data, then
    # combine on the shared date index (parts can have slightly different
    # coverage at the edges of the record).
    total_weight = sum(w for _, w in series_list)
    combined = None
    for part_series, weight in series_list:
        contribution = part_series * (weight / total_weight)
        combined = (
            contribution
            if combined is None
            else combined.add(contribution, fill_value=0)
        )
    return combined.rename("saturation_ratio")


def build_daily_saturation(region_ids):
    """
    Run the bucket model for every region and stack into one long DataFrame:
    columns [drought_region_id, measured_at, saturation_ratio].
    """
    geometries = dl.load_region_geometries(region_ids)
    frames = []
    for region_id in region_ids:
        series = compute_region_saturation(region_id, geometries[region_id])
        if series is None:
            print(f"  region {region_id}: no rainfall, skipped")
            continue
        frame = series.reset_index()
        frame.columns = ["measured_at", "saturation_ratio"]
        frame["drought_region_id"] = region_id
        frames.append(frame)
        print(f"  region {region_id}: {len(series)} days, mean S={series.mean():.3f}")
    if not frames:
        raise RuntimeError("No region produced saturation data.")
    return pd.concat(frames, ignore_index=True)


def animate_monthly_moisture(mean_by_region, output_file):
    """Animate mean monthly modelled saturation per region (see task-1 script)."""
    ids = sorted(mean_by_region["drought_region_id"].unique())
    all_geometries = dl.load_region_geometries()
    geometries = {rid: all_geometries[rid] for rid in ids if rid in all_geometries}
    label_points = {rid: dl.region_representative_point(rid) for rid in geometries}

    rm.animate_monthly_regions(
        mean_by_region[mean_by_region["drought_region_id"].isin(geometries)],
        geometries,
        region_col="drought_region_id",
        month_col="month",
        value_col="saturation_ratio",
        label_points=label_points,
        output_file=output_file,
        fps=2,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        title="Modelled mean monthly saturation",
        cbar_label="Saturation ratio S",
    )


def aggregate_bucket_moisture():
    """Drive the bucket model with historical rainfall and aggregate like the BAFU data."""
    region_ids = region_ids_from_calibration()
    print(f"Simulating {len(region_ids)} regions over {min(YEARS)}-{max(YEARS)}...")
    daily = build_daily_saturation(region_ids)
    daily["measured_at"] = pd.to_datetime(daily["measured_at"])

    # 1. Yearly aggregation
    yearly = (
        daily.groupby(["drought_region_id", daily["measured_at"].dt.year])[
            "saturation_ratio"
        ]
        .mean()
        .reset_index()
    )
    yearly.columns = ["region_id", "year", "mean_moisture"]
    yearly.to_csv(YEARLY_OUTPUT_FILE, index=False)

    # 2. Monthly aggregation (calendar month within each year)
    daily["year_month"] = daily["measured_at"].dt.to_period("M")
    monthly = (
        daily.groupby(["drought_region_id", "year_month"])["saturation_ratio"]
        .mean()
        .reset_index()
    )
    monthly.to_csv(MONTHLY_OUTPUT_FILE, index=False)

    # 3. Mean monthly saturation by region (climatology over all years)
    daily["month"] = daily["measured_at"].dt.month
    mean_monthly_by_region = (
        daily.groupby(["drought_region_id", "month"])["saturation_ratio"]
        .mean()
        .reset_index()
    )
    mean_monthly_by_region.to_csv(BY_REGION_OUTPUT_FILE, index=False)

    markers = ["o", "s", "^", "D"]

    # Static line plot of the monthly climatology
    plt.figure(figsize=(12, 6))
    for i, region_id in enumerate(mean_monthly_by_region["drought_region_id"].unique()):
        marker_custom = markers[(i // 10) % len(markers)]
        region_data = mean_monthly_by_region[
            mean_monthly_by_region["drought_region_id"] == region_id
        ]
        plt.plot(
            region_data["month"],
            region_data["saturation_ratio"],
            marker=marker_custom,
            label=f"Region {region_id}",
        )
    plt.title("Modelled Mean Monthly Saturation by Region")
    plt.xlabel("Month")
    plt.ylabel("Mean Saturation Ratio S")
    plt.xticks(range(1, 13))
    plt.grid()
    plt.legend(
        title="Region",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # outside the axes on the right
        ncol=2,  # horizontal layout
        fontsize="small",
        frameon=False,
    )
    plt.savefig(f"{OUTDIR}/mean_monthly_moisture_by_region.png", bbox_inches="tight")
    plt.close()

    # 4. Animated monthly map
    animate_monthly_moisture(
        mean_monthly_by_region, f"{OUTDIR}/mean_moisture_by_region.gif"
    )

    print("Bucket-model aggregation complete. Results saved to CSV files.")


if __name__ == "__main__":
    aggregate_bucket_moisture()
