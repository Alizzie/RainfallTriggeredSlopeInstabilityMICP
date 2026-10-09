"""
08_compute_ead.py - Expected annual landslide damage for one canton.
"""

import json
import os
import re
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import data_loader as dl

argparse = argparse.ArgumentParser()

argparse.add_argument(
    "--canton_ids",
    default=[15],
    type=lambda s: [int(x) for x in s.split(",")],
)
argparse.add_argument(
    "--canton_label",
    default="Glarus",
    help="Label for the canton",
)
argparse.add_argument(
    "--bare_name", default=None, help="Flow-Py output directory for bare"
)
argparse.add_argument(
    "--micp_name", default=None, help="Flow-Py output directory for MICP"
)
argparse.add_argument(
    "--root_name", default=None, help="Flow-Py output directory for root reinforcement"
)
argparse.add_argument(
    "--root_micp_name", default=None, help="Flow-Py output directory for root + MICP"
)
args = argparse.parse_args()
# ============================================================== CONFIG =====

CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
TARGET_CANTON_IDS = args.canton_ids
CANTON_LABEL = args.canton_label

FLOWPY_DIR = "output/flowpy/glarus"
INPUT_DIR = f"{FLOWPY_DIR}/{args.bare_name}/flowpy_input/baseline"
BASELINE_SUMMARY = (
    f"{FLOWPY_DIR}/{args.bare_name}/exposure_summary/exposure_summary.csv"
)
MICP_SUMMARY = (
    f"{FLOWPY_DIR}/{args.micp_name}/exposure_summary/exposure_summary.csv"
    if args.micp_name
    else None
)
ROOT_SUMMARY = (
    f"{FLOWPY_DIR}/{args.root_name}/exposure_summary/exposure_summary.csv"
    if args.root_name
    else None
)
ROOT_MICP_SUMMARY = (
    f"{FLOWPY_DIR}/{args.root_micp_name}/exposure_summary/exposure_summary.csv"
    if args.root_micp_name
    else None
)

COST_SUMMARY = "output/costs/cost_summary.csv"
COSTS_RAW = "output/costs/ref_costs.csv"
COST_CAP_CHF = 5_000_000

OUTDIR = f"{FLOWPY_DIR}/{args.bare_name}/ead"
os.makedirs(OUTDIR, exist_ok=True)

COL_PROCESS = "main process"
COL_DESC = "description of the event"

# Mapping between cost-table asset names and exposure-summary layer names.
# Adjust only if 01_flowpy_exposure.py uses different labels.
EXPOSURE_LAYER = {
    "road": "roads",
    "building": "buildings",
    "railway": "railway",
    "forest": "forest",
    "agricultural land": "agricultural land",
    "power supply": "power supply",
}

# Keep the same asset order everywhere.
ASSET_ORDER = [
    "road",
    "building",
    "railway",
    "agricultural land",
    "forest",
    "power supply",
]

# Description-based damage classification. This determines FREQUENCY only;
# costs come exclusively from the manually reviewed cost table.
# Patterns are intentionally auditable rather than machine-learned.
ASSET_PATTERNS = {
    "building": re.compile(
        r"\b(?:geb[aä]ude|gebauede|haus|h[aä]user|wohnhaus|wohngeb[aä]ude|"
        r"stall|scheune|garage|hotel|restaurant|h[uü]tte|liegenschaft)\w*\b",
        re.I,
    ),
    "railway": re.compile(
        r"\b(?:bahn|bahnlinie|bahntrassee|gleis|gleise|eisenbahn|sbb|rh[bB])\w*\b",
        re.I,
    ),
    "forest": re.compile(
        r"\b(?:wald|schutzwald|forst|forstweg|waldweg|waldstrasse|waldstra[sß]e)\w*\b",
        re.I,
    ),
    "agricultural land": re.compile(
        r"\b(?:landwirtschaft|kulturland|acker|wiese|weide|alpweide|flur|"
        r"rebberg|weinberg)\w*\b",
        re.I,
    ),
    "power supply": re.compile(
        r"\b(?:strom|stromleitung|elektrizit[aä]t|elektrisch|freileitung|"
        r"hochspannung|leitungsmast|strommast|transformator|trafostation)\w*\b",
        re.I,
    ),
    "road": re.compile(
        r"\b(?:strasse|stra[sß]e|kantonsstrasse|kantonsstra[sß]e|hauptstrasse|"
        r"hauptstra[sß]e|fahrbahn|fahrweg|verkehrsweg|autobahn|br[uü]cke)\w*\b",
        re.I,
    ),
}

# Terms that should count as forest infrastructure rather than ordinary road
# damage. This prevents a Forstweg from feeding both forest and road EAD.
FOREST_ROAD = re.compile(
    r"\b(?:forstweg|waldweg|forststrasse|forststra[sß]e|waldstrasse|waldstra[sß]e)\w*\b",
    re.I,
)

# =========================================================== END CONFIG ====


def load_canton():
    """Canton mask and transform on the Flow-Py grid."""
    meta_path = f"{INPUT_DIR}/meta.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    r0, r1, c0, c1 = meta["crop_window"]
    win = Window(c0, r0, c1 - c0, r1 - r0)
    with rasterio.open(CANTON_ID_TIF) as src:
        canton = src.read(1, window=win)
        transform = rasterio.windows.transform(win, src.transform)
    return np.isin(canton, TARGET_CANTON_IDS), transform


def canton_events(in_canton, transform):
    """Return landslide records whose coordinates fall inside the canton."""
    df = dl.load_wsl_inventory()
    n_all = len(df)

    df = df[df[COL_PROCESS].astype(str).str.contains("Rutschung", na=False)].copy()
    print(f"  {len(df):,}/{n_all:,} inventory events are landslides")

    h, w = in_canton.shape
    keep = []
    for x, y in zip(df["x"], df["y"]):
        r, c = rasterio.transform.rowcol(transform, x, y)
        keep.append(bool(in_canton[r, c]) if (0 <= r < h and 0 <= c < w) else False)

    out = df.loc[keep].reset_index(drop=True)
    # save new df
    outpath = f"{OUTDIR}/canton_events.csv"
    out.to_csv(outpath, index=False)
    print(f"  {len(out):,} fall inside canton {TARGET_CANTON_IDS} by coordinate")
    return out


def load_reviewed_costs(path=COST_SUMMARY):
    """Load reviewed per-asset n/p25/median/p75 values."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reviewed cost summary not found: {path}\n"
            "Run 12_plot_costs.py first so cost_summary.csv exists."
        )

    c = pd.read_csv(path)
    c.columns = [str(x).strip().lower() for x in c.columns]
    required = {"asset", "n", "p25", "median", "mean", "p75"}
    missing = required - set(c.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    c["asset"] = c["asset"].astype(str).str.strip().str.lower()
    for col in ["n", "p25", "median", "mean", "p75"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")

    c = c[c["asset"].isin(ASSET_ORDER)].copy()
    c = c.dropna(subset=["median"])
    if c.empty:
        raise ValueError(f"No recognised per-asset costs found in {path}")

    return c.set_index("asset").reindex(ASSET_ORDER).dropna(how="all")


def load_raw_reviewed_costs(path=COSTS_RAW):
    """Return one reviewed CHF observation per row for true boxplots."""
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]

    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "type":
            ren[c] = "asset"
        elif lc == "costs":
            ren[c] = "cost_chf"
    df = df.rename(columns=ren)
    if not {"asset", "cost_chf"}.issubset(df.columns):
        return None

    df["asset"] = df["asset"].astype(str).str.strip().str.lower()
    df["cost_chf"] = pd.to_numeric(
        df["cost_chf"]
        .astype(str)
        .str.replace("CHF", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    df = df[
        df["asset"].isin(ASSET_ORDER)
        & df["cost_chf"].notna()
        & (df["cost_chf"] > 0)
        & (df["cost_chf"] <= COST_CAP_CHF)
    ].copy()
    return df[["asset", "cost_chf"]]


def classify_assets(text):
    """Return all clearly named damaged assets in one WSL description."""
    text = "" if pd.isna(text) else str(text)
    hits = set()

    for asset, pat in ASSET_PATTERNS.items():
        if pat.search(text):
            hits.add(asset)

    # Forest tracks are treated as forest infrastructure, not ordinary roads.
    if FOREST_ROAD.search(text):
        hits.add("forest")
        hits.discard("road")

    return sorted(hits)


def classify_event_table(df):
    """Create an auditable event x asset table and one-row-per-event audit."""
    rows = []
    audit = []

    for idx, ev in df.iterrows():
        assets = classify_assets(ev.get(COL_DESC, ""))
        audit.append(
            {
                "event_index": idx,
                "date": ev["date"].date() if pd.notna(ev.get("date")) else None,
                "municipality": ev.get("municipality"),
                "assets": "; ".join(assets) if assets else "unclassified",
                "description": ev.get(COL_DESC, ""),
            }
        )
        for asset in assets:
            rows.append({"event_index": idx, "asset": asset})

    return pd.DataFrame(rows), pd.DataFrame(audit)


def exposure_table(path):
    """Read an exposure summary as layer -> affected."""
    if not path or not os.path.exists(path):
        return None
    x = pd.read_csv(path)
    if not {"layer", "affected"}.issubset(x.columns):
        raise ValueError(f"Exposure summary lacks layer/affected columns: {path}")
    return x.set_index("layer")["affected"].astype(float)


def build_ead(asset_events, costs, years_span):
    """Frequency x reviewed cost for every asset."""
    counts = (
        asset_events["asset"].value_counts()
        if not asset_events.empty
        else pd.Series(dtype=float)
    )

    rows = []
    for asset in ASSET_ORDER:
        if asset not in costs.index:
            continue
        n_events = int(counts.get(asset, 0))
        lam = n_events / years_span
        cr = costs.loc[asset]
        rows.append(
            {
                "asset": asset,
                "events": n_events,
                "events_per_year": lam,
                "cost_n": int(cr["n"]) if pd.notna(cr["n"]) else np.nan,
                "cost_p25": float(cr["p25"]),
                "cost_median": float(cr["median"]),
                "cost_mean": float(cr["mean"]),
                "cost_p75": float(cr["p75"]),
                "ead_p25": lam * float(cr["p25"]),
                "ead_median": lam * float(cr["median"]),
                "ead_mean": lam * float(cr["mean"]),
                "ead_p75": lam * float(cr["p75"]),
            }
        )

    return pd.DataFrame(rows).set_index("asset")


def add_treatment(ead, baseline_summary, treatment_summary, treatment):
    """Scale asset EAD using baseline -> treatment affected exposure."""
    out = ead.copy()

    exposure_col = f"exposure_{treatment}"
    ratio_col = f"exposure_ratio_{treatment}"

    out[exposure_col] = np.nan
    out[ratio_col] = np.nan

    for q in ["p25", "median", "mean", "p75"]:
        out[f"{treatment}_ead_{q}"] = np.nan

    eb = exposure_table(baseline_summary)
    et = exposure_table(treatment_summary)

    if eb is None or et is None:
        return out, False

    if "exposure_baseline" not in out.columns:
        out["exposure_baseline"] = np.nan

    any_ratio = False

    for asset in out.index:
        layer = EXPOSURE_LAYER.get(asset)

        if layer not in eb.index or layer not in et.index:
            if out.loc[asset, "ead_median"] == 0:
                out.loc[asset, ratio_col] = 1.0

                for q in ["p25", "median", "mean", "p75"]:
                    out.loc[asset, f"{treatment}_ead_{q}"] = 0.0

            continue

        baseline = float(eb.loc[layer])
        treated = float(et.loc[layer])

        if baseline <= 0:
            continue

        ratio = treated / baseline

        out.loc[asset, "exposure_baseline"] = baseline
        out.loc[asset, exposure_col] = treated
        out.loc[asset, ratio_col] = ratio

        for q in ["p25", "median", "mean", "p75"]:
            out.loc[asset, f"{treatment}_ead_{q}"] = out.loc[asset, f"ead_{q}"] * ratio

        any_ratio = True

    return out, any_ratio


def _pretty(asset):
    return {
        "road": "Road",
        "building": "Building",
        "railway": "Railway",
        "agriculture land": "Agricultural land",
        "forest": "Forest",
        "power supply": "Power supply",
    }.get(asset, asset.title())


def _boxplot_for_assets(ax, values_by_asset, order, xlabel, title, labels=None):
    """Horizontal true boxplots, one asset per row."""
    data = [np.asarray(values_by_asset[a], dtype=float) for a in order]
    positions = np.arange(1, len(order) + 1)

    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        showmeans=True,
        meanline=True,
        meanprops={"color": "#c62828", "linewidth": 1.7, "linestyle": "--"},
        medianprops={"color": "black", "linewidth": 1.7},
        boxprops={"alpha": 0.40},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={"markersize": 3.5, "alpha": 0.35},
    )
    # Let matplotlib choose its standard palette elsewhere; use neutral boxes
    # here so the figure does not imply an extra categorical variable.
    for b in bp["boxes"]:
        b.set_facecolor("0.75")

    ax.set_yticks(positions)
    ax.set_yticklabels([_pretty(a) for a in order])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    if labels:
        # Fixed annotation column: avoids labels running into the boxes.
        for pos, asset in zip(positions, order):
            ax.text(
                1.02,
                pos,
                labels[asset],
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8.5,
                clip_on=False,
            )


def plot_asset_frequency(
    ead, years_span, n_classified_events, n_canton_events, outpath
):
    """Asset-event counts and annual frequencies.

    Counts are not mutually exclusive because one event may damage several
    assets. A bar chart is therefore more honest than a pie chart.
    """
    shown = ead[ead["events"] > 0].copy()
    if shown.empty:
        return
    order = shown["events"].sort_values().index.tolist()
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9.5, 0.72 * len(order) + 2.6))
    vals = shown.loc[order, "events"].to_numpy()
    ax.barh(y, vals, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(a) for a in order])
    ax.set_xlabel("Number of Glarus landslide records naming the asset")
    ax.set_title(
        f"Asset frequency in canton {CANTON_LABEL}\n"
        rf"$\lambda_a = N_a / {years_span}$ observation years",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    xmax = max(vals) if len(vals) else 1
    ax.set_xlim(0, xmax * 1.38)
    for i, asset in enumerate(order):
        r = shown.loc[asset]
        ax.text(
            r["events"] + xmax * 0.025,
            i,
            f"{int(r['events'])} events   λ={r['events_per_year']:.3f}/yr",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        f"{n_classified_events}/{n_canton_events} records name ≥1 modelled asset. "
        "Assets overlap: one event can appear in more than one bar.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_observation_record(per_year, n_canton_events, n_classified_events, outpath):
    """Standalone time series of all recorded canton landslide events."""
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(per_year.index, per_year.values, alpha=0.85)
    mean = per_year.mean()
    ax.axhline(mean, ls="--", lw=1.7, label=f"mean = {mean:.2f} events/year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recorded landslide events")
    ax.set_title(
        f"Recorded landslide events (canton {CANTON_LABEL})",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    coverage = 100 * n_classified_events / n_canton_events if n_canton_events else 0
    fig.text(
        0.5,
        0.015,
        f"{n_classified_events}/{n_canton_events} records name ≥1 modelled asset "
        f"({coverage:.1f}% classification coverage).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_consequence_by_asset(ead, raw_costs, outpath):
    """Standalone reviewed cost distributions by asset."""
    shown = ead[ead["events"] > 0].copy()
    if shown.empty:
        return
    order = shown["cost_mean"].sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(11.2, 0.78 * len(order) + 2.8))

    if raw_costs is not None and not raw_costs.empty:
        vals = {}
        labels = {}
        for a in order:
            raw = raw_costs.loc[raw_costs["asset"] == a, "cost_chf"].values
            vals[a] = (
                raw
                if len(raw)
                else np.array(
                    [
                        shown.loc[a, "cost_p25"],
                        shown.loc[a, "cost_mean"],
                        shown.loc[a, "cost_p75"],
                    ]
                )
            )
            labels[a] = (
                f"mean CHF {shown.loc[a, 'cost_mean']:,.0f}   "
                f"median CHF {shown.loc[a, 'cost_median']:,.0f}   "
                f"n={int(shown.loc[a, 'cost_n'])}"
            )
        _boxplot_for_assets(
            ax,
            vals,
            order,
            "Reviewed damage cost per event [CHF, log scale]",
            "Reviewed damage cost by asset",
            labels=labels,
        )
    else:
        positions = np.arange(1, len(order) + 1)
        for pos, a in zip(positions, order):
            r = shown.loc[a]
            ax.barh(
                pos,
                r["cost_p75"] - r["cost_p25"],
                left=r["cost_p25"],
                height=0.5,
                alpha=0.28,
            )
            ax.vlines(r["cost_median"], pos - 0.25, pos + 0.25, color="black", lw=1.7)
            ax.vlines(
                r["cost_mean"],
                pos - 0.25,
                pos + 0.25,
                color="#c62828",
                lw=1.7,
                linestyle="--",
            )
            ax.text(
                1.02,
                pos,
                f"mean CHF {r['cost_mean']:,.0f}   "
                f"median CHF {r['cost_median']:,.0f}   n={int(r['cost_n'])}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8.5,
                clip_on=False,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty(a) for a in order])
        ax.set_xscale("log")
        ax.set_xlabel("Reviewed damage cost per event [CHF, log scale]")
        ax.set_title("Reviewed damage cost by asset", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0.05, 0.79, 1))
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_ead(
    per_year, ead, raw_costs, n_canton_events, n_classified_events, years_span, outpath
):
    """Four panels: time record -> asset frequency -> costs -> EAD."""
    shown = ead[ead["events"] > 0].copy()
    if shown.empty:
        return

    # Rank boxplots by mean EAD so consequence and risk panels line up.
    order = shown["ead_mean"].sort_values().index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))

    # 1. Observation record through time.
    ax = axes[0, 0]
    ax.bar(per_year.index, per_year.values, alpha=0.85)
    mean = per_year.mean()
    ax.axhline(mean, ls="--", lw=1.7, label=f"all landslides: {mean:.2f}/year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recorded landslide events")
    ax.set_title(
        "1. OBSERVATION RECORD\nall canton landslide records", fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    coverage = 100 * n_classified_events / n_canton_events if n_canton_events else 0
    ax.text(
        0.02,
        0.98,
        f"{n_classified_events}/{n_canton_events} name ≥1 modelled asset "
        f"({coverage:.0f}% classified)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    # 2. Asset frequency. This makes the denominator explicit.
    ax = axes[0, 1]
    freq_order = shown["events"].sort_values().index.tolist()
    yy = np.arange(len(freq_order))
    ax.barh(yy, shown.loc[freq_order, "events"].values, alpha=0.82)
    ax.set_yticks(yy)
    ax.set_yticklabels([_pretty(a) for a in freq_order])
    ax.set_xlabel("Records naming damaged asset")
    ax.set_title(
        f"2. ASSET FREQUENCY\nλ = asset-event count / {years_span} years",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    xmax = shown["events"].max()
    ax.set_xlim(0, xmax * 1.42)
    for i, asset in enumerate(freq_order):
        r = shown.loc[asset]
        ax.text(
            r["events"] + xmax * 0.025,
            i,
            f"{int(r['events'])}  |  λ={r['events_per_year']:.3f}/yr",
            va="center",
            fontsize=8.5,
        )
    ax.text(
        0.01,
        -0.16,
        f"48 classified records are coverage, not the frequency denominator. "
        f"One event may name several assets.",
        transform=ax.transAxes,
        fontsize=8.3,
    )

    # 3. Reviewed national damage costs: true boxplots when raw reviewed data exist.
    ax = axes[1, 0]
    if raw_costs is not None and not raw_costs.empty:
        vals = {
            a: raw_costs.loc[raw_costs["asset"] == a, "cost_chf"].values for a in order
        }
        # Every shown asset should have raw values, but fall back to summary if not.
        vals = {
            a: (
                v
                if len(v)
                else np.array(
                    [
                        shown.loc[a, "cost_p25"],
                        shown.loc[a, "cost_mean"],
                        shown.loc[a, "cost_p75"],
                    ]
                )
            )
            for a, v in vals.items()
        }
        labels = {
            a: (
                f"mean CHF {shown.loc[a, 'cost_mean']:,.0f}   "
                f"median CHF {shown.loc[a, 'cost_median']:,.0f}   "
                f"n={int(shown.loc[a, 'cost_n'])}"
            )
            for a in order
        }
        _boxplot_for_assets(
            ax,
            vals,
            order,
            "Reviewed damage cost per event [CHF, log scale]",
            "3. CONSEQUENCE\nreviewed national cost distributions",
            labels=labels,
        )
    else:
        # Summary-only fallback: draw IQR boxes, not fake whiskers.
        positions = np.arange(1, len(order) + 1)
        for pos, a in zip(positions, order):
            r = shown.loc[a]
            ax.barh(
                pos,
                r["cost_p75"] - r["cost_p25"],
                left=r["cost_p25"],
                height=0.5,
                alpha=0.28,
            )
            ax.vlines(r["cost_median"], pos - 0.25, pos + 0.25, color="black", lw=1.7)
            ax.vlines(
                r["cost_mean"],
                pos - 0.25,
                pos + 0.25,
                color="#c62828",
                lw=1.7,
                linestyle="--",
            )
            ax.text(
                1.02,
                pos,
                f"mean CHF {r['cost_mean']:,.0f}   "
                f"median CHF {r['cost_median']:,.0f}   n={int(r['cost_n'])}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8.5,
                clip_on=False,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty(a) for a in order])
        ax.set_xscale("log")
        ax.set_xlabel("Reviewed damage cost per event [CHF, log scale]")
        ax.set_title(
            "3. CONSEQUENCE\nIQR boxes (raw cost file not found)", fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.25)

    # 4. EAD distribution: same raw cost samples multiplied by lambda_a.
    ax = axes[1, 1]
    if raw_costs is not None and not raw_costs.empty:
        vals_ead = {}
        labels = {}
        for a in order:
            raw = raw_costs.loc[raw_costs["asset"] == a, "cost_chf"].values
            lam = shown.loc[a, "events_per_year"]
            if len(raw):
                vals_ead[a] = raw * lam
            else:
                vals_ead[a] = np.array(
                    [
                        shown.loc[a, "ead_p25"],
                        shown.loc[a, "ead_mean"],
                        shown.loc[a, "ead_p75"],
                    ]
                )
            labels[a] = f"CHF {shown.loc[a, 'ead_mean']:,.0f}/yr   " f"λ={lam:.3f}/yr"
        _boxplot_for_assets(
            ax,
            vals_ead,
            order,
            "Expected annual damage [CHF/year, log scale]",
            "4. RISK\nasset frequency x reviewed cost",
            labels=labels,
        )
    else:
        positions = np.arange(1, len(order) + 1)
        for pos, a in zip(positions, order):
            r = shown.loc[a]
            ax.barh(
                pos,
                r["ead_p75"] - r["ead_p25"],
                left=r["ead_p25"],
                height=0.5,
                alpha=0.28,
            )
            ax.vlines(r["ead_median"], pos - 0.25, pos + 0.25, color="black", lw=1.7)
            ax.vlines(
                r["ead_mean"],
                pos - 0.25,
                pos + 0.25,
                color="#c62828",
                lw=1.7,
                linestyle="--",
            )
            ax.text(
                1.02,
                pos,
                f"CHF {r['ead_mean']:,.0f}/yr   λ={r['events_per_year']:.3f}/yr",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8.5,
                clip_on=False,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty(a) for a in order])
        ax.set_xscale("log")
        ax.set_xlabel("Expected annual damage [CHF/year, log scale]")
        ax.set_title(
            "4. RISK\nIQR sensitivity (raw cost file not found)", fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.25)

    total = shown["ead_mean"].sum()
    fig.suptitle(
        f"Expected annual landslide damage (canton {CANTON_LABEL})\n"
        f"asset-specific mean-cost estimate = CHF {total:,.0f}/year",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=(0, 0.025, 0.92, 0.97))
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_ead_comparison(ead, raw_costs, has_micp, outpath):
    """Standalone asset EAD boxplots, with a clean annotation column."""
    shown = ead[ead["events"] > 0].copy()
    if shown.empty:
        return
    order = shown["ead_mean"].sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10.8, 0.78 * len(order) + 2.8))

    if raw_costs is not None and not raw_costs.empty:
        vals = {}
        labels = {}
        for a in order:
            lam = shown.loc[a, "events_per_year"]
            raw = raw_costs.loc[raw_costs["asset"] == a, "cost_chf"].values
            vals[a] = (
                raw * lam
                if len(raw)
                else np.array(
                    [
                        shown.loc[a, "ead_p25"],
                        shown.loc[a, "ead_mean"],
                        shown.loc[a, "ead_p75"],
                    ]
                )
            )
            labels[a] = (
                f"CHF {shown.loc[a, 'ead_mean']:,.0f}/yr   "
                f"N={int(shown.loc[a, 'events'])}, λ={lam:.3f}/yr"
            )
        _boxplot_for_assets(
            ax,
            vals,
            order,
            "Expected annual damage [CHF/year, log scale]",
            f"Expected annual damage by asset (canton {CANTON_LABEL})",
            labels=labels,
        )
    else:
        positions = np.arange(1, len(order) + 1)
        for pos, a in zip(positions, order):
            r = shown.loc[a]
            ax.barh(
                pos,
                r["ead_p75"] - r["ead_p25"],
                left=r["ead_p25"],
                height=0.5,
                alpha=0.28,
            )
            ax.vlines(r["ead_median"], pos - 0.25, pos + 0.25, color="black", lw=1.7)
            ax.vlines(
                r["ead_mean"],
                pos - 0.25,
                pos + 0.25,
                color="#c62828",
                lw=1.7,
                linestyle="--",
            )
            ax.text(
                1.02,
                pos,
                f"CHF {r['ead_mean']:,.0f}/yr   N={int(r['events'])}, "
                f"λ={r['events_per_year']:.3f}/yr",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8.5,
                clip_on=False,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty(a) for a in order])
        ax.set_xscale("log")
        ax.set_xlabel("Expected annual damage [CHF/year, log scale]")
        ax.set_title(
            f"Expected annual damage by asset (canton {CANTON_LABEL})",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.25)

    total = shown["ead_mean"].sum()

    # Leave space below plot and on right for annotations
    fig.tight_layout(rect=(0, 0.14, 0.80, 1))

    # Center relative to the AXES, not the whole figure
    ax.text(
        0.5,
        -0.12,
        f"Total mean-cost EAD: CHF {total:,.0f}/year.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.5,
        -0.16,
        "For each asset: EAD = λ × reviewed damage cost; "
        "λ = asset-event count / observation years.\n"
        "Black line = median, red dashed = mean (the mean is the expectation "
        "and gives the total above).",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.savefig(
        outpath,
        dpi=170,
        facecolor="white",
        bbox_inches="tight",
    )

    fig.tight_layout(rect=(0, 0.05, 0.80, 1))
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def report_treatment(log, ead, treatment, label, baseline_total):
    col = f"{treatment}_ead_mean"

    if col not in ead.columns:
        log(f"{label:36s}: not run")
        return

    valid = ead[col].notna()

    if not valid.all():
        missing = ", ".join(_pretty(a) for a in ead.index[~valid])
        log(f"{label:36s}: partial")
        log(f"  Missing exposure ratios: {missing}")
        return

    total = ead[col].sum()
    avoided = baseline_total - total
    reduction = 100 * avoided / baseline_total if baseline_total > 0 else np.nan

    log(f"{label:36s}: CHF {total:,.0f}/year")
    log(f"  Avoided EAD                        : CHF {avoided:,.0f}/year")
    log(f"  Relative reduction                : {reduction:.1f}%")


def plot_treatment_comparison(ead, outpath):
    treatments = {
        "Baseline": "ead_mean",
        "MICP": "micp_ead_mean",
        "Root": "root_ead_mean",
        "Root + MICP": "root_micp_ead_mean",
    }

    totals = {}
    for label, col in treatments.items():
        if col in ead.columns and ead[col].notna().all():
            totals[label] = ead[col].sum()

    if not totals:
        return

    labels = list(totals)
    values = list(totals.values())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(labels, values)

    ax.set_ylabel("Expected annual damage [CHF/year]")
    ax.set_title(f"Expected annual damage (canton {CANTON_LABEL})", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for i, value in enumerate(values):
        ax.text(i, value, f"CHF {value:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"-> {outpath}")


def main():
    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    in_canton, transform = load_canton()
    df = canton_events(in_canton, transform)
    if df.empty:
        log("[stop] No landslide events inside the canton. Check canton IDs/grid.")
        return

    costs = load_reviewed_costs()
    raw_costs = load_raw_reviewed_costs()

    years = df["date"].dt.year.dropna().astype(int)
    if years.empty:
        log("[stop] Canton events have no usable dates.")
        return
    year0, year1 = int(years.min()), int(years.max())
    span = year1 - year0 + 1
    per_year = (
        years.value_counts().reindex(range(year0, year1 + 1), fill_value=0).sort_index()
    )

    asset_events, audit = classify_event_table(df)
    audit.to_csv(f"{OUTDIR}/ead_event_asset_audit.csv", index=False)
    n_classified = int(audit["assets"].ne("unclassified").sum())

    ead = build_ead(asset_events, costs, span)
    ead, has_micp = add_treatment(ead, BASELINE_SUMMARY, MICP_SUMMARY, "micp")
    ead, has_root = add_treatment(ead, BASELINE_SUMMARY, ROOT_SUMMARY, "root")
    ead, has_root_micp = add_treatment(
        ead, BASELINE_SUMMARY, ROOT_MICP_SUMMARY, "root_micp"
    )

    # Main outputs.
    ead.reset_index().to_csv(f"{OUTDIR}/ead_by_asset.csv", index=False)
    # Standalone figures: one question per plot.
    plot_observation_record(
        per_year,
        n_canton_events=len(df),
        n_classified_events=n_classified,
        outpath=f"{OUTDIR}/observation_record.png",
    )
    plot_asset_frequency(
        ead,
        years_span=span,
        n_classified_events=n_classified,
        n_canton_events=len(df),
        outpath=f"{OUTDIR}/asset_frequency.png",
    )
    plot_consequence_by_asset(
        ead,
        raw_costs,
        outpath=f"{OUTDIR}/consequence_by_asset.png",
    )
    plot_ead_comparison(ead, raw_costs, has_micp, f"{OUTDIR}/ead_by_asset.png")
    plot_treatment_comparison(ead, f"{OUTDIR}/treatment_comparison.png")

    # Report.
    log("")
    log("=" * 76)
    log(f"EXPECTED ANNUAL DAMAGE (canton {CANTON_LABEL})")
    log("=" * 76)
    log(f"Observation period                 : {year0}-{year1} ({span} years)")
    log(f"Canton landslide records           : {len(df)}")
    log(
        f"Records naming >=1 modelled asset  : {n_classified} ({100*n_classified/len(df):.1f}%)"
    )
    log("")
    log("ASSET-SPECIFIC EAD")
    log("  asset               events   lambda/yr   mean cost       EAD/yr")
    for asset, r in ead.iterrows():
        log(
            f"  {_pretty(asset):18s} {int(r['events']):6d}   {r['events_per_year']:8.3f}   "
            f"CHF {r['cost_mean']:>10,.0f}   CHF {r['ead_mean']:>10,.0f}"
        )

    total_med = ead["ead_median"].sum()
    total_lo = ead["ead_p25"].sum()
    total_hi = ead["ead_p75"].sum()
    total_mean = ead["ead_mean"].sum()

    log("")
    log(f"TOTAL EAD (mean cost)              : CHF {total_mean:,.0f}/year")
    log(f"  median-cost floor                : CHF {total_med:,.0f}/year")
    log(
        f"  cost-IQR sensitivity             : CHF {total_lo:,.0f} - {total_hi:,.0f}/year"
    )
    log("                                     (frequency held fixed; not a CI)")

    log("")
    log("TREATMENT COMPARISON")

    if has_micp:
        report_treatment(log, ead, "micp", "MICP", total_mean)
    else:
        log("MICP                                : not run")

    if has_root:
        report_treatment(log, ead, "root", "Root reinforcement", total_mean)
    else:
        log("Root reinforcement                  : not run")

    if has_root_micp:
        report_treatment(log, ead, "root_micp", "Root + MICP", total_mean)
    else:
        log("Root + MICP                         : not run")

    log("")
    log("IMPORTANT INTERPRETATION")
    log("  - Each asset uses its own reviewed national damage-cost distribution.")
    log(
        f"  - Asset frequency is lambda_a = N_asset / {span} observation years; 48 classified events is coverage, not the denominator."
    )
    log("  - Event descriptions determine N_asset; inspect ead_event_asset_audit.csv.")
    log(
        "  - An event may count for multiple assets if several damaged assets are named."
    )
    log("  - Forest tracks are assigned to forest, not ordinary road frequency.")
    log(
        "  - Unclassified events do not enter the asset-specific total, so the total is conservative"
    )
    log("    with respect to description-based asset attribution.")
    log(
        "  - The cost p25-p75 range is sensitivity to consequence cost only, not full uncertainty."
    )
    log(
        "  - MICP scaling assumes asset EAD changes in proportion to affected exposure for that asset."
    )
    log(
        "  - Subtype medians are not used in EAD because subtype classification/sample sizes remain sparse."
    )

    with open(f"{OUTDIR}/ead_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n-> {OUTDIR}/ead_by_asset.csv")
    print(f"-> {OUTDIR}/ead_event_asset_audit.csv")
    print(f"-> {OUTDIR}/ead_report.txt")


if __name__ == "__main__":
    main()
