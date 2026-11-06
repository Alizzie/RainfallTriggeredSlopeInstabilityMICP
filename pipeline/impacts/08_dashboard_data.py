"""07_export_dashboard_json.py

Writes landslide-model.json for the interactive canton map, from the real model
instead of the dummy file.

What it produces
----------------
1. Per canton and per soil-saturation level S (bucket-model degree of
   saturation), the same four FoS fields as 01_fos_maps.py:
   bare soil, roots, MICP (+MICP_C kPa), roots + MICP.
   The pore-pressure ratio is derived exactly as in the model:
       m_pp = max(0, (S - S_onset) / (1 - S_onset))      (physics.pore_pressure_ratio)
   with S_onset taken from locked_params.json.
2. hydrology.rain_response_curve: antecedent rainfall over ANTECEDENT_DAYS days
   -> bucket saturation S. Built empirically, like 06_rainfall_baseline.py:
   the calibrated bucket model is run on MeteoSwiss RhiresD rainfall for every
   drought region, then the ANTECEDENT_DAYS-day rolling rainfall is binned and
   the median S per bin is stored. The map's "Use live rainfall" button sums
   Open-Meteo precipitation over the same window and reads S from this curve.

Put this file next to 01_fos_maps.py (it imports constants and helpers from it,
so MICP_C, ROOT_C, the rasters and the FoS maths stay a single source of truth).

Run from the repository root:
    python pipeline/impacts/07_export_dashboard_json.py
    python pipeline/impacts/07_export_dashboard_json.py --window-days 3   # compare windows
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const, data_loader as dl, physics
from pipeline.utils import constants as auct

# 01_fos_maps.py starts with a digit, so it cannot be imported with `import`.
_spec = importlib.util.spec_from_file_location(
    "fos_maps", Path(__file__).with_name("01_fos_maps.py")
)
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ANTECEDENT_DAYS = 7  # 7 d beat 3 d in your test; override with --window-days
RAIN_START_YEAR = 1991
CURVE_YEARS = range(RAIN_START_YEAR, 2024)  # years without a rainfall file are skipped
SPINUP_DAYS = 365  # discard the first year of every bucket run
MIN_BIN_SAMPLES = 300  # a rain bin needs at least this many days to enter the curve
RAIN_BIN_EDGES = [0, 1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 300, 1000]

SAT_STEP = 0.04  # grid spacing above S_onset; the scale starts AT S_onset
LEGACY_M_PP = [
    0.12,
    0.35,
    0.50,
    0.79,
    1.00,
]  # published design values, kept as exact levels
CHECK_M_PP = 0.79  # level used for the sanity printout against your published numbers
TARGET_FOS = 1.0  # fm.required_cohesion() is the cohesion needed for FoS = 1

LOCKED_PARAMS_PATH = Path("output/temporal/01_calibrate_saturation/locked_params.json")
OUTDIR = Path("output/dashboard")
OUT_JSON = OUTDIR / "landslide-model.json"

# Canton table: (id, name, lat, lon). The numbers here are NOT used for matching:
# canton_names.csv / canton_id_25m.tif use their own numbering, so cantons are matched
# to the raster values BY NAME in resolve_cantons(). lat/lon is the single reference
# point per canton used for the live-rainfall query.
_CANTON_TABLE = {
    1: ("ZH", "Zürich", 47.41, 8.65),
    2: ("BE", "Bern", 46.82, 7.62),
    3: ("LU", "Luzern", 47.08, 8.14),
    4: ("UR", "Uri", 46.77, 8.63),
    5: ("SZ", "Schwyz", 47.05, 8.76),
    6: ("OW", "Obwalden", 46.85, 8.25),
    7: ("NW", "Nidwalden", 46.93, 8.40),
    8: ("GL", "Glarus", 46.98, 9.07),
    9: ("ZG", "Zug", 47.15, 8.52),
    10: ("FR", "Fribourg", 46.72, 7.11),
    11: ("SO", "Solothurn", 47.35, 7.62),
    12: ("BS", "Basel-Stadt", 47.56, 7.60),
    13: ("BL", "Basel-Landschaft", 47.45, 7.72),
    14: ("SH", "Schaffhausen", 47.71, 8.60),
    15: ("AR", "Appenzell Ausserrhoden", 47.38, 9.28),
    16: ("AI", "Appenzell Innerrhoden", 47.32, 9.42),
    17: ("SG", "St. Gallen", 47.25, 9.32),
    18: ("GR", "Graubünden", 46.66, 9.63),
    19: ("AG", "Aargau", 47.39, 8.15),
    20: ("TG", "Thurgau", 47.57, 9.09),
    21: ("TI", "Ticino", 46.30, 8.80),
    22: ("VD", "Vaud", 46.62, 6.55),
    23: ("VS", "Valais", 46.20, 7.60),
    24: ("NE", "Neuchâtel", 46.99, 6.80),
    25: ("GE", "Genève", 46.20, 6.14),
    26: ("JU", "Jura", 47.35, 7.15),
}

# Filled by resolve_cantons(): raster value -> (id, name, lat, lon).
CANTONS = {}
N_IDS = 0  # index 0 = outside any canton

ALIASES = {
    "genf": "geneve",
    "freiburg": "fribourg",
    "neuenburg": "neuchatel",
    "tessin": "ticino",
    "waadt": "vaud",
    "wallis": "valais",
    "graubunden": "graubunden",
    "grigioni": "graubunden",
}


# ----------------------------------------------------------------------------
# Hydrology: rainfall window -> bucket saturation
# ----------------------------------------------------------------------------
def load_s_onset():
    if not LOCKED_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{LOCKED_PARAMS_PATH} not found. Run the onset calibration first."
        )
    with open(LOCKED_PARAMS_PATH, "r", encoding="utf-8") as handle:
        return float(json.load(handle)["s_onset"])


def collect_rain_saturation_pairs(s_onset, window_days):
    """Daily (antecedent rainfall, bucket saturation) pairs for every drought region."""
    if not os.path.exists(dl.PATH_CALIB):
        raise FileNotFoundError(f"Calibration file missing: {dl.PATH_CALIB}")
    region_ids = sorted(pd.read_csv(dl.PATH_CALIB)["region_id"].unique())

    frames = []
    for region_id in region_ids:
        easting, northing = dl.get_region_coordinates(int(region_id))
        rain = dl.load_rainfall(easting, northing, CURVE_YEARS)
        drainage, et, et_amp = dl.load_calibration_params(int(region_id))
        if rain is None or rain.empty or drainage is None:
            print(f"  region {region_id}: skipped (no rainfall or calibration)")
            continue

        # The bucket function prints a block of text per call; keep the log readable.
        with contextlib.redirect_stdout(io.StringIO()):
            saturation = physics.calculate_daily_saturation(
                rain.to_numpy(dtype=float),
                n=const.N,
                n_perp=const.H_PERP,
                m0=s_onset,
                s_pp_onset=s_onset,
                drainage_rate=drainage,
                et_rate=et,
                day_of_year=rain.index.dayofyear.to_numpy(),
                et_amplitude=et_amp,
            )
        window_sum = rain.rolling(f"{window_days}D").sum()
        frame = pd.DataFrame(
            {"rain": window_sum.to_numpy(dtype=float), "sat": saturation},
            index=rain.index,
        ).iloc[SPINUP_DAYS:]
        frames.append(frame)
        print(f"  region {region_id}: {len(frame):,} days")

    if not frames:
        raise RuntimeError("No rainfall/saturation series could be built.")
    return pd.concat(frames, ignore_index=True)


def build_rain_response_curve(pairs):
    """Median saturation per antecedent-rainfall bin, forced to be non-decreasing."""
    rain = pairs["rain"].to_numpy(dtype=float)
    sat = pairs["sat"].to_numpy(dtype=float)
    ok = np.isfinite(rain) & np.isfinite(sat)
    rain, sat = rain[ok], sat[ok]

    points = []
    for lo, hi in zip(RAIN_BIN_EDGES[:-1], RAIN_BIN_EDGES[1:]):
        in_bin = (rain >= lo) & (rain < hi)
        if in_bin.sum() < MIN_BIN_SAMPLES:
            continue
        points.append(
            {
                "mm": float(np.median(rain[in_bin])),
                "s_median": float(np.median(sat[in_bin])),
                "s_p10": float(np.quantile(sat[in_bin], 0.10)),
                "s_p90": float(np.quantile(sat[in_bin], 0.90)),
                "n": int(in_bin.sum()),
            }
        )
    if len(points) < 3:
        raise RuntimeError("Too few rainfall bins with enough samples for a curve.")

    mm = np.array([p["mm"] for p in points])
    keep = np.concatenate([[True], np.diff(mm) > 0])  # strictly increasing mm
    points = [p for p, k in zip(points, keep) if k]
    s_monotone = np.maximum.accumulate(
        np.clip([p["s_median"] for p in points], 0.0, 1.0)
    )

    curve = [
        [round(p["mm"], 1), round(float(s), 3)] for p, s in zip(points, s_monotone)
    ]
    spread = [
        {
            "mm": round(p["mm"], 1),
            "s_p10": round(p["s_p10"], 3),
            "s_p90": round(p["s_p90"], 3),
            "n": p["n"],
        }
        for p in points
    ]
    return curve, spread


# ----------------------------------------------------------------------------
# Spatial inputs
# ----------------------------------------------------------------------------
def _fold(text):
    text = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in text.lower() if ch.isalpha())


def resolve_cantons():
    """Match canton_names.csv (raster value -> name) to the table above by name."""
    global CANTONS, N_IDS
    names = dl.load_canton_names()
    if not names:
        raise FileNotFoundError(
            "canton_names.csv not found; cannot map raster values to cantons."
        )
    by_name = {_fold(v[1]): v for v in _CANTON_TABLE.values()}
    resolved, missing = {}, []
    for num, csv_name in names.items():
        key = _fold(csv_name)
        key = ALIASES.get(key, key)
        entry = by_name.get(key)
        if entry is None:
            missing.append(f"{num}={csv_name}")
        else:
            resolved[int(num)] = entry
    if missing:
        raise ValueError(
            f"No match for CSV cantons: {missing}. Add them to ALIASES / the table."
        )
    absent = {v[0] for v in _CANTON_TABLE.values()} - {v[0] for v in resolved.values()}
    if absent:
        print(f"WARNING: cantons not present in the CSV: {sorted(absent)}")
    CANTONS = resolved
    N_IDS = max(CANTONS) + 1
    for num in sorted(CANTONS):
        print(f"  raster value {num:2d} -> {CANTONS[num][0]} {CANTONS[num][1]}")


def load_pixels():
    """Valid soil pixels (same mask as 01_fos_maps.py) with their canton id."""
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta = src.read(1).astype(np.float32)
        nodata = src.nodata
        px_km2 = float(src.res[0] * src.res[1]) / 1e6

    forest, _, _ = dl.load_forest_raster()
    if forest.shape != beta.shape:
        raise ValueError(f"forest shape {forest.shape} != slope shape {beta.shape}")
    nonsoil = fm.read_aligned(fm.NONSOIL_TIF, beta.shape, "non-soil")
    micp = fm.read_aligned(fm.MICP_FEASIBILITY_TIF, beta.shape, "MICP feasibility")
    canton_id, _, _ = dl.load_canton_raster()
    if canton_id.shape != beta.shape:
        raise ValueError(f"canton raster {canton_id.shape} != slope {beta.shape}")

    valid = np.isfinite(beta)
    if nodata is not None:
        valid &= beta != nodata
    valid &= (beta >= fm.BETA_MIN) & (beta <= fm.BETA_MAX)
    valid &= nonsoil != 1

    ids_all = canton_id.astype(np.int64).ravel()
    ids_all = np.where((ids_all >= 1) & (ids_all < N_IDS), ids_all, 0)
    canton_px = np.bincount(ids_all, minlength=N_IDS)
    cid = ids_all.reshape(canton_id.shape)[valid]

    return pack_pixels(
        beta[valid], forest[valid], micp[valid] == 1, cid, canton_px, px_km2
    )


def pack_pixels(beta, forest, micp, cid, canton_px, px_km2):
    """Bundle valid-pixel arrays and the fixed cohesion fields."""
    return {
        "beta": beta,
        "c_root": fm.cohesion_from(forest, fm.ROOT_C),
        "micp_feasible": micp,
        "c_micp": np.where(micp, fm.MICP_C, 0.0).astype(np.float32),
        "cid": cid,
        "n_valid_by_canton": np.bincount(cid, minlength=N_IDS),
        "canton_px": canton_px,
        "px_km2": px_km2,
    }


# ----------------------------------------------------------------------------
# FoS per saturation level
# ----------------------------------------------------------------------------
def saturation_levels(s_onset):
    grid = np.arange(s_onset, 1.0 - 1e-9, SAT_STEP)
    legacy = s_onset + np.array(LEGACY_M_PP) * (1.0 - s_onset)
    levels = np.unique(np.round(np.concatenate([grid, [1.0, s_onset], legacy]), 6))
    return [float(s) for s in levels if s_onset - 1e-9 <= s <= 1.0]


def _count(mask, cid):
    return np.bincount(cid[mask], minlength=N_IDS)


def evaluate_level(px, saturation, s_onset):
    """Per-canton counts for one saturation level, plus national shares."""
    m_pp = float(physics.pore_pressure_ratio(saturation, s_onset))
    beta, cid = px["beta"], px["cid"]

    fos_bare = fm.fos_field(beta, m_pp, fm.BARE_C)
    fos_root = fm.fos_field(beta, m_pp, fm.BARE_C + px["c_root"])
    fos_micp = fm.fos_field(beta, m_pp, fm.BARE_C + px["c_micp"])
    fos_both = fm.fos_field(beta, m_pp, fm.BARE_C + px["c_root"] + px["c_micp"])
    c_req = fm.required_cohesion(beta, m_pp)

    u_bare = fos_bare < 1
    # More cohesion can only stabilise, so each treated set is a subset of u_bare;
    # the "&" makes that explicit and keeps saved + residual == unstable exactly.
    u_root = (fos_root < 1) & u_bare
    u_micp = (fos_micp < 1) & u_bare
    u_both = (fos_both < 1) & u_bare

    n_unstable = _count(u_bare, cid)
    n_valid = px["n_valid_by_canton"]
    fos_sum = np.bincount(cid, weights=fos_bare.astype(np.float64), minlength=N_IDS)
    creq_sum = np.bincount(
        cid[u_bare], weights=c_req[u_bare].astype(np.float64), minlength=N_IDS
    )

    counts = {
        "valid": n_valid,
        "unstable": n_unstable,
        "residual_micp": _count(u_micp, cid),
        "residual_roots": _count(u_root, cid),
        "residual_roots_micp": _count(u_both, cid),
        "micp_feasible_unstable": _count(u_bare & px["micp_feasible"], cid),
        "mean_fos": np.where(n_valid > 0, fos_sum / np.maximum(n_valid, 1), np.nan),
        "mean_creq": np.where(
            n_unstable > 0, creq_sum / np.maximum(n_unstable, 1), 0.0
        ),
    }

    total_valid = max(int(n_valid.sum()), 1)
    total_unstable = max(int(u_bare.sum()), 1)
    national = {
        "m_pp": m_pp,
        "unstable_bare_pct": 100.0 * u_bare.sum() / total_valid,
        "unstable_roots_pct": 100.0 * u_root.sum() / total_valid,
        "unstable_micp_pct": 100.0 * u_micp.sum() / total_valid,
        "unstable_roots_micp_pct": 100.0 * u_both.sum() / total_valid,
        "stabilised_roots_pct": 100.0 * (u_bare & ~u_root).sum() / total_unstable,
        "stabilised_micp_pct": 100.0 * (u_bare & ~u_micp).sum() / total_unstable,
        "stabilised_roots_micp_pct": 100.0 * (u_bare & ~u_both).sum() / total_unstable,
        "micp_feasible_share_pct": 100.0
        * (u_bare & px["micp_feasible"]).sum()
        / total_unstable,
    }
    return m_pp, counts, national


def canton_row(num, saturation, m_pp, counts, px):
    """One scenario row for one canton, in the schema the map reads."""
    px_km2 = px["px_km2"]
    area = px["canton_px"][num] * px_km2
    unstable = counts["unstable"][num] * px_km2
    res_micp = counts["residual_micp"][num] * px_km2
    res_roots = counts["residual_roots"][num] * px_km2
    res_both = counts["residual_roots_micp"][num] * px_km2
    saved = unstable - res_micp
    feasible = counts["micp_feasible_unstable"][num] * px_km2
    mean_fos = counts["mean_fos"][num]
    return {
        "saturation": round(saturation, 6),
        "m_pp": round(m_pp, 4),
        "unstable_area_km2": round(unstable, 2),
        "unstable_fraction": round(unstable / area, 4) if area > 0 else 0.0,
        "mean_fos": None if not np.isfinite(mean_fos) else round(float(mean_fos), 3),
        "required_cohesion_kpa": round(float(counts["mean_creq"][num]), 1),
        "area_saved_km2": round(saved, 2),
        "residual_area_km2": round(res_micp, 2),
        "protection_ratio": round(saved / unstable, 3) if unstable > 0 else 0.0,
        "residual_roots_km2": round(res_roots, 2),
        "residual_roots_micp_km2": round(res_both, 2),
        "area_saved_roots_km2": round(unstable - res_roots, 2),
        "area_saved_roots_micp_km2": round(unstable - res_both, 2),
        "micp_feasible_km2": round(feasible, 2),
        "micp_feasible_share": round(feasible / unstable, 3) if unstable > 0 else 0.0,
    }


def build_cantons(px, levels, s_onset):
    cantons = {}
    for num, (cid, name, lat, lon) in CANTONS.items():
        cantons[cid] = {
            "id": cid,
            "name": name,
            "lat": lat,
            "lon": lon,
            "area_km2": round(float(px["canton_px"][num] * px["px_km2"]), 1),
            "modelled_soil_km2": round(
                float(px["n_valid_by_canton"][num] * px["px_km2"]), 1
            ),
            "scenarios": [],
        }

    national_at = []
    for index, saturation in enumerate(levels, start=1):
        m_pp, counts, national = evaluate_level(px, saturation, s_onset)
        national_at.append((saturation, national))
        for num, (cid, *_rest) in CANTONS.items():
            cantons[cid]["scenarios"].append(
                canton_row(num, saturation, m_pp, counts, px)
            )
        print(f"  level {index}/{len(levels)}: S={saturation:.3f}  m_pp={m_pp:.3f}")
    return cantons, national_at


def self_check(cantons):
    """Internal consistency of the numbers that go into the file."""
    problems = 0
    for cid, canton in cantons.items():
        previous = None
        for row in canton["scenarios"]:
            u = row["unstable_area_km2"]
            ordered = (
                row["residual_roots_micp_km2"] <= row["residual_roots_km2"] + 0.011
                and row["residual_roots_micp_km2"] <= row["residual_area_km2"] + 0.011
                and row["residual_area_km2"] <= u + 0.011
                and row["residual_roots_km2"] <= u + 0.011
                and abs(row["area_saved_km2"] + row["residual_area_km2"] - u) <= 0.011
            )
            if not ordered or u > canton["area_km2"] + 0.011:
                print(f"WARNING {cid} S={row['saturation']}: inconsistent areas {row}")
                problems += 1
            if previous is not None and u + 0.011 < previous:
                print(
                    f"WARNING {cid} S={row['saturation']}: unstable area falls as S rises"
                )
                problems += 1
            previous = u
    print(f"self-check: {problems} problem(s)")
    return problems


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def assemble_model(
    cantons, levels, s_onset, curve, spread, feasible_share, window_days, px
):
    return {
        "meta": {
            "title": "Rainfall-induced landslide susceptibility, Swiss cantons",
            "model_version": "1.0",
            "generated": date.today().isoformat(),
            "source": "Infinite-slope FoS model (core.physics) on a 25 m grid, aggregated per canton.",
            "calibration": "Bucket model with per-region drainage/ET calibration; S_onset from locked_params.json.",
            "target_fos": TARGET_FOS,
            "micp_cohesion_gain_kpa": fm.MICP_C,
            "micp_feasible_share": round(feasible_share, 2),
            "micp_feasible_share_basis": (
                "Percent of initially unstable (bare-soil FoS < 1) soil pixels, "
                f"at m_pp = {CHECK_M_PP}, that lie in MICP-feasible areas."
            ),
            "s_onset": s_onset,
            "saturation_levels": levels,
            "saturation_definition": (
                "Bucket-model degree of saturation S. Pore-pressure ratio "
                "m_pp = max(0, (S - s_onset) / (1 - s_onset))."
            ),
            "slope_range_deg": [fm.BETA_MIN, fm.BETA_MAX],
            "pixel_area_km2": px["px_km2"],
            "units": {"area": "km2", "cohesion": "kPa", "precipitation": "mm"},
            "field_definitions": {
                "unstable_area_km2": "Modelled soil with FoS < 1, bare soil (baseline).",
                "residual_area_km2": "Still unstable with MICP only.",
                "residual_roots_km2": "Still unstable with roots only.",
                "residual_roots_micp_km2": "Still unstable with roots + MICP.",
                "area_saved_km2": "unstable_area_km2 - residual_area_km2 (MICP only).",
                "area_saved_roots_km2": "unstable_area_km2 - residual_roots_km2 (roots only).",
                "area_saved_roots_micp_km2": "unstable_area_km2 - residual_roots_micp_km2 (roots + MICP).",
                "micp_feasible_km2": "Part of the canton's unstable area where MICP injection is feasible.",
                "micp_feasible_share": "micp_feasible_km2 / unstable_area_km2 (0-1), per canton and scenario.",
                "required_cohesion_kpa": "Mean extra cohesion over the unstable pixels needed for FoS = 1.",
                "mean_fos": "Mean bare-soil FoS over the modelled soil pixels of the canton; null if there are none.",
                "area_km2": "Whole canton. Areas above count only modelled soil on 15-35 degree slopes.",
            },
        },
        "hydrology": {
            "antecedent_window_h": int(window_days * 24),
            "note": (
                f"Median bucket-model saturation per {window_days}-day antecedent rainfall bin "
                "(MeteoSwiss RhiresD, all drought regions). Interpolate linearly between points."
            ),
            "rain_response_curve": curve,
            "rain_response_spread": spread,
        },
        "cantons": cantons,
    }


def print_national_check(national_at):
    saturation, nat = min(
        national_at, key=lambda item: abs(item[1]["m_pp"] - CHECK_M_PP)
    )
    print(
        f"\nNational check at S={saturation:.3f} (m_pp={nat['m_pp']:.3f}); compare with your 01_fos_maps summary:"
    )
    print("  unstable share of modelled soil pixels")
    print(f"    bare soil     {nat['unstable_bare_pct']:6.1f} %")
    print(f"    roots         {nat['unstable_roots_pct']:6.1f} %")
    print(f"    MICP          {nat['unstable_micp_pct']:6.1f} %")
    print(f"    roots + MICP  {nat['unstable_roots_micp_pct']:6.1f} %")
    print("  share of bare-unstable pixels stabilised")
    print(f"    roots         {nat['stabilised_roots_pct']:6.1f} %")
    print(f"    MICP          {nat['stabilised_micp_pct']:6.1f} %")
    print(f"    roots + MICP  {nat['stabilised_roots_micp_pct']:6.1f} %")
    print(
        f"  MICP-feasible share of unstable pixels: {nat['micp_feasible_share_pct']:.2f} %"
    )
    return nat


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-days", type=int, default=ANTECEDENT_DAYS)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    s_onset = load_s_onset()
    print(
        f"S_onset = {s_onset}   MICP cohesion = {fm.MICP_C} kPa   window = {args.window_days} d"
    )

    print("\nCanton matching (raster value -> canton, by name)")
    resolve_cantons()

    print("\n1/3  Rain -> saturation curve")
    pairs = collect_rain_saturation_pairs(s_onset, args.window_days)
    curve, spread = build_rain_response_curve(pairs)
    print(f"  curve points (mm, S): {curve}")

    print("\n2/3  FoS per canton and saturation level")
    px = load_pixels()
    total_km2 = px["canton_px"][1:].sum() * px["px_km2"]
    print(f"  cantons cover {total_km2:,.0f} km2 (Switzerland is about 41,285 km2)")
    levels = saturation_levels(s_onset)
    cantons, national_at = build_cantons(px, levels, s_onset)
    problems = self_check(cantons)

    nat = print_national_check(national_at)

    print("\n3/3  Writing JSON")
    model = assemble_model(
        cantons,
        levels,
        s_onset,
        curve,
        spread,
        nat["micp_feasible_share_pct"],
        args.window_days,
        px,
    )
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved {OUT_JSON}")

    worst = max(
        (row["unstable_fraction"], cid, row["saturation"])
        for cid, canton in cantons.items()
        for row in canton["scenarios"]
    )
    print(
        f"Largest canton share unstable: {worst[0]:.1%} ({worst[1]}, S={worst[2]:.2f}). "
        "Set riskDomain in LANDSLIDE_MAP_CONFIG near this value if the map saturates to red."
    )
    if problems:
        print("Fix the warnings above before publishing the file.")


if __name__ == "__main__":
    main()
