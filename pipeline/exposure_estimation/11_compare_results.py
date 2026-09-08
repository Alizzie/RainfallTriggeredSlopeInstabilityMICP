"""
11_compare_scenarios.py - Compare every (treatment x m_pp) run in one place.

Reads the outputs the pipeline has already written for each scenario
directory `output/flowpy/glarus/<treatment>_<m_pp>/` and produces:

    scenario_matrix.csv        one tidy row per scenario, every headline number
    scenario_matrix_wide.csv   the same, pivoted treatment x m_pp per metric
    consistency_report.txt     grid checks + monotonicity checks + summary

    grid_fos.png               FoS map            rows = m_pp, cols = treatment
    grid_footprint.png         runout corridors   rows = m_pp, cols = treatment
    grid_risk.png              CHF/year per cell  rows = m_pp, cols = treatment
    grid_exposure_bars.png     % of asset hit     rows = m_pp, cols = treatment

    curve_release.png          release cells vs m_pp, one line per treatment
    curve_footprint.png        affected canton area vs m_pp
    curve_exposure.png         % affected per asset, small multiples
    curve_ead.png              total EAD vs m_pp, one line per treatment
    heatmap_reduction.png      % EAD avoided vs bare, m_pp x treatment
    ranking_shift.png          do the top-ranked segments move under treatment?

Grids are for MAPS. Scalars get curves, because the finding is the shape of
the curve (MICP's effect shrinking as m_pp rises), and a 5x4 grid of numbers
hides exactly that.

USAGE
    python 11_compare_scenarios.py
    python 11_compare_scenarios.py --m_pp 0.12,0.35,0.5,0.79,1.0
    python 11_compare_scenarios.py --only grid_risk,curve_ead --stride 3
"""

import json
import os
import sys
import argparse
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch

# ============================================================== PARSER =====

parser = argparse.ArgumentParser()
parser.add_argument(
    "--m_pp",
    default="0.12,0.35,0.5,0.79,1.0",
    help="pore-pressure ratios, in the exact spelling used in directory names",
)
parser.add_argument(
    "--treatments",
    default="bare,root,micp,root_micp",
    help="treatment scenarios, in the order they should appear as columns",
)
parser.add_argument(
    "--base_dir",
    default="output/flowpy/glarus",
    help="parent directory holding the scenario directories",
)
parser.add_argument(
    "--name_template",
    default="{treatment}_{m_pp}",
    help="how scenario directory names are built",
)
parser.add_argument(
    "--fos_template",
    default="output/impacts/01_maps/m_pp_{m_pp}/fos_{treatment}_{m_pp}.tif",
    help="where the FoS raster for a scenario lives (national grid)",
)
parser.add_argument("--canton_ids", default="15")
parser.add_argument("--canton_label", default="Glarus")
parser.add_argument(
    "--outdir",
    default="output/flowpy/glarus/_comparison",
    help="where the comparison outputs go",
)
parser.add_argument(
    "--stride",
    type=int,
    default=2,
    help="decimate rasters by this factor for the grid figures only",
)
parser.add_argument(
    "--only",
    default=None,
    help="comma-separated subset of figures to build (default: all)",
)
args = parser.parse_args()

# ============================================================== CONFIG =====

M_PPS = [s.strip() for s in args.m_pp.split(",") if s.strip()]
TREATMENTS = [s.strip() for s in args.treatments.split(",") if s.strip()]
BASE_DIR = args.base_dir
TARGET_CANTON_IDS = [int(x) for x in str(args.canton_ids).split(",")]
CANTON_LABEL = args.canton_label
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
CELL_SIZE_M = 25.0
CELL_AREA_M2 = CELL_SIZE_M**2

# Column in ead_by_asset.csv holding each treatment's EAD.
# NOTE the longest-prefix match: "root_micp" must not resolve to "root".
EAD_COL = {
    "bare": "ead_median",
    "root": "root_ead_median",
    "micp": "micp_ead_median",
    "root_micp": "root_micp_ead_median",
}

# Exposure layers to carry into the comparison, in reporting order.
EXPOSURE_LAYERS = [
    "protection forest",
    "forest",
    "buildings",
    "railway",
    "roads",
    "agricultural land",
    "protected areas",
]

TREATMENT_LABEL = {
    "bare": "Bare soil",
    "root": "Root cohesion",
    "micp": "MICP",
    "root_micp": "Root + MICP",
}

TREATMENT_COLOR = {
    "bare": "#37474f",
    "root": "#2e7d32",
    "micp": "#1565c0",
    "root_micp": "#c62828",
}

# Ranking-shift comparison: how close two cluster centres must be to count as
# the same location.
RANK_MATCH_M = 150.0
RANK_TOP_N = 20

FIGURES = [
    "grid_fos",
    "grid_footprint",
    "grid_risk",
    "grid_exposure_bars",
    "curve_release",
    "curve_footprint",
    "curve_exposure",
    "curve_ead",
    "heatmap_reduction",
    "ranking_shift",
]
WANTED = [s.strip() for s in args.only.split(",")] if args.only else list(FIGURES)

# =========================================================== END CONFIG ====


def label_m_pp(m):
    return f"m_pp = {m}"


def label_treatment(t):
    return TREATMENT_LABEL.get(t, t.replace("_", " ").title())


def colour(t):
    return TREATMENT_COLOR.get(t, "#616161")


def as_float(m):
    try:
        return float(m)
    except ValueError:
        return np.nan


# ------------------------------------------------------------ scenario paths


def resolve_dir(treatment, m_pp):
    """Scenario directory, tolerating 1 / 1.0 / 1.00 spellings of m_pp."""
    candidates = [m_pp]
    f = as_float(m_pp)
    if np.isfinite(f):
        candidates += [f"{f:g}", f"{f:.1f}", f"{f:.2f}"]
    seen = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.append(cand)
        name = args.name_template.format(treatment=treatment, m_pp=cand)
        path = os.path.join(BASE_DIR, name)
        if os.path.isdir(path):
            return name, path
    name = args.name_template.format(treatment=treatment, m_pp=m_pp)
    return name, os.path.join(BASE_DIR, name)


def resolve_fos(treatment, m_pp):
    candidates = [m_pp]
    f = as_float(m_pp)
    if np.isfinite(f):
        candidates += [f"{f:g}", f"{f:.1f}", f"{f:.2f}"]
    for cand in dict.fromkeys(candidates):
        p = args.fos_template.format(treatment=treatment, m_pp=cand)
        if os.path.exists(p):
            return p
    return args.fos_template.format(treatment=treatment, m_pp=m_pp)


class Scenario:
    """Everything the comparison needs to know about one run."""

    def __init__(self, treatment, m_pp):
        self.treatment = treatment
        self.m_pp = m_pp
        self.m_pp_f = as_float(m_pp)
        self.name, self.dir = resolve_dir(treatment, m_pp)
        self.input_dir = f"{self.dir}/flowpy_input/baseline"
        self.meta_path = f"{self.input_dir}/meta.json"
        self.z_delta = f"{self.dir}/z_delta.tif"
        self.cell_counts = f"{self.dir}/cell_counts.tif"
        self.release = f"{self.input_dir}/release.tif"
        self.dem = f"{self.input_dir}/dem.tif"
        self.exposure_csv = f"{self.dir}/exposure_summary/exposure_summary.csv"
        self.risk_tif = f"{self.dir}/risk_ranking/risk_chf_per_year.tif"
        self.risk_by_asset = f"{self.dir}/risk_ranking/risk_by_asset.csv"
        self.top_segments = f"{self.dir}/risk_ranking/top_segments.csv"
        self.hit_rate = f"{self.dir}/validation/wsl_hit_rate.csv"
        self.fos = resolve_fos(treatment, m_pp)
        # EAD is computed once per m_pp and stored under the bare scenario.
        _, bare_dir = resolve_dir("bare", m_pp)
        self.ead_csv = f"{bare_dir}/ead/ead_by_asset.csv"

    @property
    def exists(self):
        return os.path.exists(self.z_delta) and os.path.exists(self.meta_path)

    def meta(self):
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, encoding="utf-8") as fh:
            return json.load(fh)


def build_scenarios():
    scen = {}
    for t, m in product(TREATMENTS, M_PPS):
        s = Scenario(t, m)
        scen[(t, m)] = s
    return scen


# --------------------------------------------------------------- shared grid


def load_reference(scenarios):
    """DEM, canton mask and transform, taken from the first complete run.

    Every scenario shares these by construction (the crop window in
    01_export_release_raster.py is keyed to the canton, not to the FoS map).
    That is checked explicitly in check_grids().
    """
    ref = next((s for s in scenarios.values() if s.exists), None)
    if ref is None:
        raise SystemExit(
            "No scenario directory contains z_delta.tif + meta.json.\n"
            f"Looked under {BASE_DIR} for {TREATMENTS} x {M_PPS}."
        )

    with rasterio.open(ref.z_delta) as src:
        transform = src.transform
        profile = src.profile.copy()
        shape = (src.height, src.width)

    with rasterio.open(ref.dem) as src:
        dem = src.read(1).astype(np.float32)
    dem = np.where(dem <= -9000, np.nan, dem)

    r0, r1, c0, c1 = ref.meta()["crop_window"]
    win = Window(c0, r0, c1 - c0, r1 - r0)
    with rasterio.open(CANTON_ID_TIF) as src:
        canton = src.read(1, window=win)
    in_canton = np.isin(canton, TARGET_CANTON_IDS)

    return dem, in_canton, transform, profile, shape, win, ref


def hillshade(dem):
    filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(filled, vert_exag=1.5, dx=CELL_SIZE_M, dy=CELL_SIZE_M)


def dec(arr, stride=None):
    """Decimate for display only."""
    s = args.stride if stride is None else stride
    return arr[::s, ::s] if s > 1 else arr


# ------------------------------------------------------------------ harvest


def read_csv_safe(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - a bad CSV must not kill the sweep
        print(f"  [warn] could not read {path}: {exc}")
        return None


def footprint_stats(scen, in_canton):
    """Affected area, inside the canton and in the whole modelled window."""
    if not os.path.exists(scen.z_delta):
        return {}
    with rasterio.open(scen.z_delta) as src:
        z = src.read(1).astype(np.float32)
    fp = np.isfinite(z) & (z > 0)

    out = {
        "affected_km2_window": fp.sum() * CELL_AREA_M2 / 1e6,
        "affected_km2_canton": (fp & in_canton).sum() * CELL_AREA_M2 / 1e6,
        "pct_canton_affected": 100 * (fp & in_canton).sum() / max(in_canton.sum(), 1),
    }

    if os.path.exists(scen.cell_counts):
        with rasterio.open(scen.cell_counts) as src:
            counts = src.read(1).astype(np.float32)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        inside = counts[fp & in_canton]
        if inside.size:
            out["median_paths"] = float(np.median(inside))
            out["p90_paths"] = float(np.percentile(inside, 90))
            out["max_paths"] = float(inside.max())
            # Total convergence is the quantity risk is split by, so it is the
            # honest scalar for "how much routing did treatment remove".
            out["sum_paths_canton"] = float(inside.sum())
    return out


def ead_total(scen):
    """Total median-cost EAD for this treatment, from the shared EAD table."""
    df = read_csv_safe(scen.ead_csv)
    if df is None:
        return {}
    col = EAD_COL.get(scen.treatment)
    if col is None or col not in df.columns:
        print(f"  [warn] {scen.name}: column '{col}' missing in {scen.ead_csv}")
        return {}
    vals = pd.to_numeric(df[col], errors="coerce")
    if not vals.notna().any():
        return {}
    out = {"ead_total_chf_yr": float(vals.sum(skipna=True))}
    if "asset" in df.columns:
        for asset, v in zip(df["asset"].astype(str), vals):
            if pd.notna(v):
                out[f"ead_{asset.strip().lower().replace(' ', '_')}"] = float(v)
    return out


def exposure_row(scen):
    df = read_csv_safe(scen.exposure_csv)
    if df is None or "layer" not in df.columns:
        return {}
    df = df.set_index(df["layer"].astype(str).str.strip())
    out = {}
    for layer in EXPOSURE_LAYERS:
        if layer not in df.index:
            continue
        key = layer.replace(" ", "_")
        out[f"pct_{key}"] = float(df.loc[layer, "pct_affected"])
        out[f"affected_{key}"] = float(df.loc[layer, "affected"])
        out[f"total_{key}"] = float(df.loc[layer, "total"])
    return out


def risk_row(scen):
    df = read_csv_safe(scen.risk_by_asset)
    if df is None or "risk_raster_sum_chf_per_year" not in df.columns:
        return {}
    return {
        "risk_allocated_chf_yr": float(
            pd.to_numeric(df["risk_raster_sum_chf_per_year"], errors="coerce").sum()
        )
    }


def validation_row(scen):
    """Hit rate at radius 0 against the road control.

    Reported as a FOOTPRINT DIAGNOSTIC, not as model skill: a treated
    scenario has a smaller footprint and therefore necessarily a lower hit
    rate. Validation belongs to the calibrated baseline only.
    """
    df = read_csv_safe(scen.hit_rate)
    if df is None or "radius_m" not in df.columns:
        return {}
    sub = df[(df["radius_m"] == 0)]
    if sub.empty:
        return {}
    road = sub[sub["control_type"] == "roads"]
    row = road.iloc[0] if not road.empty else sub.iloc[0]
    return {
        "wsl_hit_rate_r0": float(row["damage_hit_rate"]),
        "wsl_control_hit_rate_r0": float(row["control_hit_rate"]),
        "wsl_lift_r0": float(row["lift"]) if pd.notna(row.get("lift")) else np.nan,
        "n_damage": int(row["n_damage"]) if pd.notna(row.get("n_damage")) else np.nan,
    }


def harvest(scenarios, in_canton):
    rows = []
    for (t, m), s in scenarios.items():
        if not s.exists:
            print(f"  [missing] {s.name}")
            rows.append(
                {
                    "treatment": t,
                    "m_pp": m,
                    "m_pp_value": s.m_pp_f,
                    "scenario": s.name,
                    "present": False,
                }
            )
            continue

        print(f"  [read]    {s.name}")
        meta = s.meta()
        row = {
            "treatment": t,
            "m_pp": m,
            "m_pp_value": s.m_pp_f,
            "scenario": s.name,
            "present": True,
            "n_unstable_cells": meta.get("n_unstable_cells"),
            "n_release_cells": meta.get("n_release_cells"),
            "unstable_area_km2": meta.get("unstable_area_km2"),
            "fos_source": meta.get("fos_source"),
        }
        row.update(footprint_stats(s, in_canton))
        row.update(exposure_row(s))
        row.update(ead_total(s))
        row.update(risk_row(s))
        row.update(validation_row(s))
        rows.append(row)

    df = pd.DataFrame(rows)
    order_t = {t: i for i, t in enumerate(TREATMENTS)}
    order_m = {m: i for i, m in enumerate(M_PPS)}
    df = df.sort_values(
        by=["m_pp", "treatment"],
        key=lambda s: s.map(order_m if s.name == "m_pp" else order_t),
    ).reset_index(drop=True)
    return df


def add_deltas(df):
    """Everything relative to bare at the same m_pp."""
    if "bare" not in TREATMENTS:
        return df
    base = df[df["treatment"] == "bare"].set_index("m_pp")

    def rel(row, col):
        if col not in df.columns or row["m_pp"] not in base.index:
            return np.nan
        b = base.loc[row["m_pp"], col]
        v = row.get(col)
        if not np.isfinite(b) or b == 0 or not np.isfinite(v):
            return np.nan
        return 100 * (b - v) / b

    for col, out in [
        ("n_release_cells", "release_reduction_pct"),
        ("affected_km2_canton", "footprint_reduction_pct"),
        ("sum_paths_canton", "convergence_reduction_pct"),
        ("ead_total_chf_yr", "ead_reduction_pct"),
    ]:
        if col in df.columns:
            df[out] = df.apply(lambda r, c=col: rel(r, c), axis=1)

    if "ead_total_chf_yr" in df.columns:
        df["ead_avoided_chf_yr"] = df.apply(
            lambda r: (
                base.loc[r["m_pp"], "ead_total_chf_yr"] - r["ead_total_chf_yr"]
                if r["m_pp"] in base.index
                else np.nan
            ),
            axis=1,
        )
    return df


# -------------------------------------------------------------- grid figures


def grid_figure(panel_fn, title, outpath, panel_size=(3.1, 3.1), cbar_fn=None):
    """rows = m_pp, cols = treatment. panel_fn(ax, scenario) draws one cell."""
    nrow, ncol = len(M_PPS), len(TREATMENTS)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(panel_size[0] * ncol + 1.4, panel_size[1] * nrow + 1.4),
        squeeze=False,
    )

    handle = None
    for i, m in enumerate(M_PPS):
        for j, t in enumerate(TREATMENTS):
            ax = axes[i][j]
            s = SCENARIOS[(t, m)]
            try:
                h = panel_fn(ax, s)
                handle = h if h is not None else handle
            except Exception as exc:  # noqa: BLE001
                ax.text(
                    0.5,
                    0.5,
                    f"failed\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#b71c1c",
                    transform=ax.transAxes,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(label_treatment(t), fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(label_m_pp(m), fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    if cbar_fn is not None and handle is not None:
        cbar_fn(fig, handle, axes)
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def panel_fos(ax, s):
    """FoS, diverging about the failure threshold."""
    if not os.path.exists(s.fos):
        ax.text(0.5, 0.5, "no FoS raster", ha="center", va="center", fontsize=8)
        return None
    r0, r1, c0, c1 = REF_META["crop_window"]
    with rasterio.open(s.fos) as src:
        fos = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    fos = np.where(np.isfinite(fos), fos, np.nan)
    fos = np.where(IN_CANTON, fos, np.nan)

    ax.imshow(dec(SHADE), cmap="gray", vmin=0, vmax=2.0)
    im = ax.imshow(
        np.ma.masked_invalid(dec(np.clip(fos, 0, 2.0))),
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0),
        interpolation="nearest",
    )
    n_unstable = int(np.nansum(fos < 1.0))
    ax.text(
        0.02,
        0.02,
        f"{n_unstable:,} unstable cells",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
    )
    return im


def panel_footprint(ax, s):
    """Runout corridors over hillshade, with release cells underneath."""
    if not os.path.exists(s.z_delta):
        ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=8)
        return None
    with rasterio.open(s.z_delta) as src:
        z = src.read(1).astype(np.float32)
    fp = np.isfinite(z) & (z > 0)

    ax.imshow(dec(SHADE), cmap="gray", vmin=0, vmax=2.0)
    sub = dec(fp)
    ax.imshow(
        np.ma.masked_where(~sub, np.ones(sub.shape, dtype=np.uint8)),
        cmap=ListedColormap(["#ff6d00"]),
        alpha=0.55,
        interpolation="nearest",
    )
    if os.path.exists(s.release):
        with rasterio.open(s.release) as src:
            rel = dec(src.read(1) > 0)
        ax.imshow(
            np.ma.masked_where(~rel, np.ones(rel.shape, dtype=np.uint8)),
            cmap=ListedColormap(["#1a237e"]),
            alpha=0.75,
            interpolation="nearest",
        )
    ax.contour(
        dec(IN_CANTON).astype(float), levels=[0.5], colors="black", linewidths=0.8
    )
    pct = 100 * (fp & IN_CANTON).sum() / max(IN_CANTON.sum(), 1)
    ax.text(
        0.02,
        0.02,
        f"{pct:.0f}% of canton",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
    )
    return None


def panel_risk(ax, s):
    """Allocated CHF/year per cell, on a shared log scale."""
    if not os.path.exists(s.risk_tif):
        ax.text(0.5, 0.5, "no risk raster", ha="center", va="center", fontsize=8)
        return None
    with rasterio.open(s.risk_tif) as src:
        risk = src.read(1).astype(np.float32)
    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)

    ax.imshow(dec(SHADE), cmap="gray", vmin=0, vmax=2.0)
    shown = dec(risk)
    m = shown > 0
    if not m.any():
        return None
    im = ax.imshow(
        np.ma.masked_where(~m, shown),
        cmap="YlOrRd",
        norm=LogNorm(vmin=RISK_VMIN, vmax=RISK_VMAX),
        interpolation="nearest",
    )
    ax.text(
        0.02,
        0.02,
        f"CHF {risk.sum():,.0f}/yr",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
    )
    return im


def panel_exposure_bars(ax, s):
    """Per-panel bar chart: % of each asset inside a corridor."""
    df = read_csv_safe(s.exposure_csv)
    if df is None or "pct_affected" not in df.columns:
        ax.text(0.5, 0.5, "no exposure summary", ha="center", va="center", fontsize=8)
        return None
    df = df.dropna(subset=["pct_affected"])
    df["layer"] = df["layer"].astype(str).str.strip()
    df = df.set_index("layer").reindex(EXPOSURE_LAYERS).dropna(subset=["pct_affected"])
    if df.empty:
        return None

    y = np.arange(len(df))[::-1]
    ax.barh(y, df["pct_affected"], color="#d84315", alpha=0.9, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlim(0, EXPOSURE_XMAX)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for yy, v in zip(y, df["pct_affected"]):
        ax.text(v + EXPOSURE_XMAX * 0.02, yy, f"{v:.0f}", va="center", fontsize=6.5)
    return None


def grid_exposure_bars_figure(outpath):
    """Same grid layout, but bars need visible axes, so it does not use
    grid_figure()."""
    nrow, ncol = len(M_PPS), len(TREATMENTS)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.6 * ncol + 1.0, 2.4 * nrow + 1.4), squeeze=False
    )
    for i, m in enumerate(M_PPS):
        for j, t in enumerate(TREATMENTS):
            ax = axes[i][j]
            panel_exposure_bars(ax, SCENARIOS[(t, m)])
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(label_treatment(t), fontsize=11, fontweight="bold")
            if i == nrow - 1:
                ax.set_xlabel("% of asset in a corridor", fontsize=8)
            if j == 0:
                ax.text(
                    -0.55,
                    0.5,
                    label_m_pp(m),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    fig.suptitle(
        f"Exposure to modelled runout, canton {CANTON_LABEL}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# ------------------------------------------------------------- curve figures


def curve(df, ycol, ylabel, title, outpath, pct=False, logy=False, note=None):
    sub = df[df["present"]] if "present" in df.columns else df
    if ycol not in sub.columns or not sub[ycol].notna().any():
        print(f"  [skip] {outpath}: no data for {ycol}")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for t in TREATMENTS:
        g = sub[sub["treatment"] == t].sort_values("m_pp_value")
        g = g[g[ycol].notna()]
        if g.empty:
            continue
        ax.plot(
            g["m_pp_value"],
            g[ycol],
            "o-",
            lw=2.2,
            ms=6,
            color=colour(t),
            label=label_treatment(t),
        )

    ax.set_xlabel(r"pore-pressure ratio $m_{pp}$")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if pct:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(title, fontsize=12, fontweight="bold")
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def curve_exposure_figure(df, outpath):
    """One small multiple per asset layer; line per treatment."""
    cols = [
        (f"pct_{l.replace(' ', '_')}", l)
        for l in EXPOSURE_LAYERS
        if f"pct_{l.replace(' ', '_')}" in df.columns
    ]
    cols = [(c, l) for c, l in cols if df[c].notna().any()]
    if not cols:
        print(f"  [skip] {outpath}: no exposure columns")
        return

    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, sharex=True
    )
    for k, (col, layer) in enumerate(cols):
        ax = axes[k // ncol][k % ncol]
        for t in TREATMENTS:
            g = df[(df["treatment"] == t) & df[col].notna()].sort_values("m_pp_value")
            if g.empty:
                continue
            ax.plot(
                g["m_pp_value"],
                g[col],
                "o-",
                lw=2,
                ms=5,
                color=colour(t),
                label=label_treatment(t),
            )
        ax.set_title(layer, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        if k % ncol == 0:
            ax.set_ylabel("% affected")
        if k // ncol == nrow - 1:
            ax.set_xlabel(r"$m_{pp}$")

    for k in range(len(cols), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TREATMENTS), fontsize=9)
    fig.suptitle(
        f"Asset exposure across scenarios, canton {CANTON_LABEL}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def heatmap_reduction(df, outpath):
    """% EAD avoided relative to bare at the same m_pp."""
    if "ead_reduction_pct" not in df.columns:
        print(f"  [skip] {outpath}: no EAD reduction column")
        return
    treatments = [t for t in TREATMENTS if t != "bare"]
    if not treatments:
        return

    mat = np.full((len(M_PPS), len(treatments)), np.nan)
    for i, m in enumerate(M_PPS):
        for j, t in enumerate(treatments):
            sel = df[(df["m_pp"] == m) & (df["treatment"] == t)]
            if not sel.empty:
                mat[i, j] = sel["ead_reduction_pct"].iloc[0]

    if not np.isfinite(mat).any():
        print(f"  [skip] {outpath}: all values missing")
        return

    fig, ax = plt.subplots(figsize=(1.9 * len(treatments) + 3.2, 0.85 * len(M_PPS) + 3))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=max(vmax, 1e-6))
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels([label_treatment(t) for t in treatments])
    ax.set_yticks(range(len(M_PPS)))
    ax.set_yticklabels([label_m_pp(m) for m in M_PPS])

    for i in range(len(M_PPS)):
        for j in range(len(treatments)):
            v = mat[i, j]
            ax.text(
                j,
                i,
                "—" if not np.isfinite(v) else f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    fig.colorbar(im, ax=ax, label="EAD avoided vs bare soil [%]", shrink=0.8)
    ax.set_title(
        "Expected annual damage avoided, relative to bare soil\n"
        "at the same pore-pressure ratio",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# ------------------------------------------------------------ ranking shift


def ranking_shift(outpath, csv_path):
    """Do the top-ranked segments survive treatment, or does the map reorder?

    Two numbers per (treatment, m_pp):
      retained  - share of bare's top-N locations still in the treatment top-N
                  (matched by centre within RANK_MATCH_M)
      risk_kept - treatment risk on those retained locations / bare risk on them

    A high retained share means treatment scales the map down without changing
    priorities. A low one means treatment moves the problem, which is a
    different and more interesting claim.
    """
    rows = []
    for m in M_PPS:
        base = read_csv_safe(SCENARIOS[("bare", m)].top_segments)
        if base is None or base.empty:
            continue
        b = base.head(RANK_TOP_N)
        for t in TREATMENTS:
            if t == "bare":
                continue
            cur = read_csv_safe(SCENARIOS[(t, m)].top_segments)
            if cur is None or cur.empty:
                continue
            c = cur.head(RANK_TOP_N)
            matched, kept_risk, base_risk = 0, 0.0, 0.0
            for _, br in b.iterrows():
                d = np.hypot(
                    c["x_lv95"].to_numpy() - br["x_lv95"],
                    c["y_lv95"].to_numpy() - br["y_lv95"],
                )
                base_risk += float(br["risk_chf_per_year"])
                if d.size and d.min() <= RANK_MATCH_M:
                    matched += 1
                    kept_risk += float(c.iloc[int(np.argmin(d))]["risk_chf_per_year"])
            rows.append(
                {
                    "m_pp": m,
                    "m_pp_value": as_float(m),
                    "treatment": t,
                    "top_n": len(b),
                    "retained": matched,
                    "retained_pct": 100 * matched / max(len(b), 1),
                    "risk_kept_pct": (
                        100 * kept_risk / base_risk if base_risk > 0 else np.nan
                    ),
                }
            )

    if not rows:
        print(f"  [skip] {outpath}: no top_segments.csv found")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"-> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for ax, col, title in [
        (
            axes[0],
            "retained_pct",
            f"Bare-soil top-{RANK_TOP_N} locations still ranked after treatment",
        ),
        (axes[1], "risk_kept_pct", "Risk remaining on those same locations"),
    ]:
        for t in TREATMENTS:
            if t == "bare":
                continue
            g = df[df["treatment"] == t].sort_values("m_pp_value")
            if g.empty:
                continue
            ax.plot(
                g["m_pp_value"],
                g[col],
                "o-",
                lw=2.2,
                ms=6,
                color=colour(t),
                label=label_treatment(t),
            )
        ax.set_xlabel(r"$m_{pp}$")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, fontweight="bold")
    axes[0].legend()
    fig.suptitle(
        f"Does treatment rescale the risk map or reorder it? (canton {CANTON_LABEL})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# --------------------------------------------------------------- consistency


def check_grids(log):
    """Every scenario must share a crop window, or nothing here is comparable."""
    log("")
    log("GRID CONSISTENCY")
    windows = {}
    for (t, m), s in SCENARIOS.items():
        if not s.exists:
            continue
        meta = s.meta()
        key = (tuple(meta.get("crop_window", [])), tuple(meta.get("shape", [])))
        windows.setdefault(key, []).append(s.name)

    if len(windows) <= 1:
        log(f"  all {sum(len(v) for v in windows.values())} runs share one grid")
    else:
        log("  [FAIL] scenarios do not share a grid - comparisons are invalid:")
        for key, names in windows.items():
            log(f"    crop_window={key[0]} shape={key[1]}")
            log(f"      {', '.join(sorted(names))}")


def check_monotonic(df, log):
    """Physical sanity: more cohesion cannot create release cells, and more
    water cannot remove them."""
    log("")
    log("MONOTONICITY")

    ok = True
    for m in M_PPS:
        sub = df[(df["m_pp"] == m) & df["n_release_cells"].notna()]
        vals = dict(zip(sub["treatment"], sub["n_release_cells"]))
        for weaker, stronger in [
            ("bare", "root"),
            ("bare", "micp"),
            ("root", "root_micp"),
            ("micp", "root_micp"),
        ]:
            if weaker in vals and stronger in vals and vals[stronger] > vals[weaker]:
                ok = False
                log(
                    f"  [FAIL] {label_m_pp(m)}: {stronger} has more release cells "
                    f"({vals[stronger]:,.0f}) than {weaker} ({vals[weaker]:,.0f}) "
                    "- check which FoS map was exported"
                )

    for t in TREATMENTS:
        sub = df[(df["treatment"] == t) & df["n_release_cells"].notna()]
        sub = sub.sort_values("m_pp_value")
        v = sub["n_release_cells"].to_numpy()
        if v.size > 1 and np.any(np.diff(v) < 0):
            ok = False
            log(
                f"  [FAIL] {label_treatment(t)}: release cells fall as m_pp rises "
                "- wetter terrain cannot be more stable"
            )

    if ok:
        log("  release-cell counts are monotone in both cohesion and m_pp")


def check_conservation(df, log):
    """09 conserves the EAD it is handed. If not, the split lost mass."""
    log("")
    log("RISK ALLOCATION")
    sub = df[df["ead_total_chf_yr"].notna() & df["risk_allocated_chf_yr"].notna()]
    if sub.empty:
        log("  [skip] EAD and/or allocated risk not available")
        return
    for _, r in sub.iterrows():
        ratio = r["risk_allocated_chf_yr"] / r["ead_total_chf_yr"]
        flag = "" if 0.98 <= ratio <= 1.02 else "   <- unallocated EAD"
        log(
            f"  {r['scenario']:<16s} EAD CHF {r['ead_total_chf_yr']:>12,.0f}/yr "
            f"-> allocated {ratio:6.1%}{flag}"
        )
    log("  (a ratio below 1 means some asset EAD had no reached cell to sit on;")
    log("   that share is real risk the map cannot place, not risk removed.)")


# --------------------------------------------------------------------- main


def main():
    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    log("=" * 76)
    log(f"SCENARIO COMPARISON - canton {CANTON_LABEL}")
    log("=" * 76)
    log(f"  treatments : {', '.join(TREATMENTS)}")
    log(f"  m_pp       : {', '.join(M_PPS)}")
    log(f"  base dir   : {BASE_DIR}")
    log("")

    df = harvest(SCENARIOS, IN_CANTON)
    df = add_deltas(df)

    df.to_csv(f"{OUTDIR}/scenario_matrix.csv", index=False)
    print(f"\n-> {OUTDIR}/scenario_matrix.csv")

    # Wide form: one block per metric, treatment x m_pp.
    metrics = [
        c
        for c in [
            "n_release_cells",
            "affected_km2_canton",
            "pct_canton_affected",
            "sum_paths_canton",
            "ead_total_chf_yr",
            "ead_reduction_pct",
            "risk_allocated_chf_yr",
        ]
        + [f"pct_{l.replace(' ', '_')}" for l in EXPOSURE_LAYERS]
        if c in df.columns
    ]
    wide = []
    for metric in metrics:
        p = df.pivot_table(
            index="treatment", columns="m_pp", values=metric, aggfunc="first"
        )
        p = p.reindex(index=[t for t in TREATMENTS if t in p.index])
        p = p.reindex(columns=[m for m in M_PPS if m in p.columns])
        p.insert(0, "metric", metric)
        wide.append(p.reset_index())
    if wide:
        pd.concat(wide, ignore_index=True).to_csv(
            f"{OUTDIR}/scenario_matrix_wide.csv", index=False
        )
        print(f"-> {OUTDIR}/scenario_matrix_wide.csv")

    check_grids(log)
    check_monotonic(df, log)
    check_conservation(df, log)

    # ---- figures ----
    print("")
    if "grid_fos" in WANTED:
        grid_figure(
            panel_fos,
            f"Factor of safety by treatment and pore-pressure ratio ({CANTON_LABEL})",
            f"{OUTDIR}/grid_fos.png",
            cbar_fn=lambda fig, h, axes: fig.colorbar(
                h,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                shrink=0.45,
                pad=0.02,
                label="FoS (clipped at 2; green = stable, red = FoS < 1)",
            ),
        )
    if "grid_footprint" in WANTED:
        grid_figure(
            panel_footprint,
            f"Modelled runout corridors ({CANTON_LABEL})\n"
            "blue = release cells, orange = reachable",
            f"{OUTDIR}/grid_footprint.png",
        )
    if "grid_risk" in WANTED:
        grid_figure(
            panel_risk,
            f"Allocated expected annual damage ({CANTON_LABEL})",
            f"{OUTDIR}/grid_risk.png",
            cbar_fn=lambda fig, h, axes: fig.colorbar(
                h,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                shrink=0.45,
                pad=0.02,
                label="CHF/year per 25 m cell (shared log scale)",
            ),
        )
    if "grid_exposure_bars" in WANTED:
        grid_exposure_bars_figure(f"{OUTDIR}/grid_exposure_bars.png")

    if "curve_release" in WANTED:
        curve(
            df,
            "n_release_cells",
            "release cells (FoS < 1, inside the canton)",
            "Unstable area across scenarios",
            f"{OUTDIR}/curve_release.png",
            note="This is the cheapest treatment signal in the pipeline: it "
            "needs the release export only, not Flow-Py.",
        )
    if "curve_footprint" in WANTED:
        curve(
            df,
            "pct_canton_affected",
            "% of canton inside a runout corridor",
            "Susceptibility envelope across scenarios",
            f"{OUTDIR}/curve_footprint.png",
            pct=True,
            note="Union of paths from every unstable cell - a susceptibility "
            "envelope, not one event's footprint.",
        )
    if "curve_exposure" in WANTED:
        curve_exposure_figure(df, f"{OUTDIR}/curve_exposure.png")
    if "curve_ead" in WANTED:
        curve(
            df,
            "ead_total_chf_yr",
            "expected annual damage [CHF/year]",
            f"Expected annual damage across scenarios ({CANTON_LABEL})",
            f"{OUTDIR}/curve_ead.png",
            note="Frequency is empirical and fixed; treatments enter through "
            "the exposure ratio in 08_compute_ead.py.",
        )
    if "heatmap_reduction" in WANTED:
        heatmap_reduction(df, f"{OUTDIR}/heatmap_reduction.png")
    if "ranking_shift" in WANTED:
        ranking_shift(f"{OUTDIR}/ranking_shift.png", f"{OUTDIR}/ranking_shift.csv")

    # ---- headline table in the report ----
    log("")
    log("HEADLINE NUMBERS")
    log(
        f"  {'scenario':<16s} {'release':>10s} {'% canton':>9s} "
        f"{'EAD CHF/yr':>13s} {'vs bare':>9s}"
    )
    for _, r in df.iterrows():
        if not r.get("present"):
            log(f"  {r['scenario']:<16s} {'-- not run --':>10s}")
            continue
        log(
            f"  {r['scenario']:<16s} "
            f"{r.get('n_release_cells', np.nan):>10,.0f} "
            f"{r.get('pct_canton_affected', np.nan):>8.1f}% "
            f"{r.get('ead_total_chf_yr', np.nan):>13,.0f} "
            f"{r.get('ead_reduction_pct', np.nan):>8.1f}%"
        )

    log("")
    log("READ THIS BEFORE QUOTING ANY NUMBER")
    log("  - The m_pp ladder is a wetness percentile ladder for this canton,")
    log("    not a set of round numbers. Say which percentile each one is.")
    log("  - Validation (04) and the Schwanden case (05) belong to the")
    log("    calibrated baseline only. A treated scenario has a smaller")
    log("    footprint, so its WSL hit rate falls by construction; that is a")
    log("    footprint diagnostic, not a loss of model skill.")
    log("  - Treatment effect enters EAD through affected EXTENT. Extent")
    log("    saturates (a corridor narrowing from 200 paths to 20 still counts")
    log("    as affected), so these reductions are a lower bound. The")
    log("    convergence_reduction_pct column is the less conservative reading.")

    with open(f"{OUTDIR}/consistency_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {OUTDIR}/consistency_report.txt")


def _risk_scale():
    """One log scale for every risk panel, or the grid is not comparable."""
    lo, hi = np.inf, -np.inf
    for s in SCENARIOS.values():
        if not os.path.exists(s.risk_tif):
            continue
        with rasterio.open(s.risk_tif) as src:
            r = src.read(1).astype(np.float32)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size:
            lo = min(lo, float(r.min()))
            hi = max(hi, float(r.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1e-3, 1.0
    return max(lo, 1e-3), hi


def _exposure_xmax():
    vals = []
    for s in SCENARIOS.values():
        d = read_csv_safe(s.exposure_csv)
        if d is not None and "pct_affected" in d.columns:
            vals += list(pd.to_numeric(d["pct_affected"], errors="coerce").dropna())
    return min(105.0, max(vals) * 1.25) if vals else 100.0


def setup():
    """Populate the shared state every panel function reads.

    Done here rather than at import so that --help costs nothing.
    """
    global SCENARIOS, DEM, IN_CANTON, TRANSFORM, REF, REF_META, SHADE
    global RISK_VMIN, RISK_VMAX, EXPOSURE_XMAX

    SCENARIOS = build_scenarios()
    DEM, IN_CANTON, TRANSFORM, _, _, _, REF = load_reference(SCENARIOS)
    REF_META = REF.meta()
    SHADE = hillshade(DEM)
    RISK_VMIN, RISK_VMAX = _risk_scale()
    EXPOSURE_XMAX = _exposure_xmax()


if __name__ == "__main__":
    setup()
    main()
