"""
10_treatment_points.py
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage

argparse = argparse.ArgumentParser()
argparse.add_argument(
    "--flowpy_name",
    default="bare_1",
    help="Flow-Py output directory",
)
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
args = argparse.parse_args()

# =================================================================== CONFIG

FLOWPY_DIR = f"output/flowpy/glarus/{args.flowpy_name}"
INPUT_DIR = f"{FLOWPY_DIR}/flowpy_input/baseline"
RISK_DIR = f"{FLOWPY_DIR}/risk_ranking"

TARGET_ID_TIF = f"{RISK_DIR}/top_risk_targets_id.tif"
TARGET_CSV = f"{RISK_DIR}/top_risk_targets.csv"

OUTDIR = f"{FLOWPY_DIR}/target_treatment"
INFRA_DIR = f"{OUTDIR}/infra_targets"
BACKTRACE_ROOT = f"{OUTDIR}/target_backtraces"
FIG_DIR = f"{OUTDIR}/figures"
PER_TARGET_FIG_DIR = f"{FIG_DIR}/per_target"

for d in [OUTDIR, INFRA_DIR, BACKTRACE_ROOT, FIG_DIR, PER_TARGET_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# How many highest-risk targets to analyse individually.
TARGET_TOP_N = 10

CELL_SIZE_M = 25.0
CELL_AREA_M2 = CELL_SIZE_M**2

# MICP injectability.
MICP_INJECTION_TIF = "data/injection/injection_map.tif"
MICP_INJECTABLE_VALUE = 3

# Canton.
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
TARGET_CANTON_IDS = args.canton_ids
CANTON_LABEL = args.canton_label

# Flow-Py.
FLOWPY_CODE_DIR = "Flowpy"
FLOWPY_CONDA_ENV = "flowpy"
FLOWPY_MAIN = "main.py"
FLOWPY_ALPHA = 25
FLOWPY_EXP = 75
FLOWPY_EXTRA_ARGS = []

AUTO_RUN_TARGET_BACKTRACES = True
REUSE_EXISTING_TARGET_BACKTRACES = True
FORCE_RERUN_TARGET_BACKTRACES = False

# ================================================================= END CONFIG


def crop_window():
    with open(f"{INPUT_DIR}/meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    r0, r1, c0, c1 = meta["crop_window"]
    return Window(c0, r0, c1 - c0, r1 - r0)


def load_inputs():
    with rasterio.open(f"{INPUT_DIR}/release.tif") as src:
        release = src.read(1) > 0
        profile = src.profile
        transform = src.transform

    with rasterio.open(f"{INPUT_DIR}/dem.tif") as src:
        dem = src.read(1).astype(np.float32)
    dem = np.where(dem <= -9000, np.nan, dem)

    with rasterio.open(TARGET_ID_TIF) as src:
        target_ids = src.read(1).astype(np.int32)

    if target_ids.shape != release.shape:
        raise ValueError(
            f"Target-ID raster shape {target_ids.shape} != release shape {release.shape}"
        )

    targets = pd.read_csv(TARGET_CSV)
    needed = {"rank", "asset_type", "asset", "risk_chf_per_year", "x_lv95", "y_lv95"}
    missing = needed - set(targets.columns)
    if missing:
        raise ValueError(f"{TARGET_CSV} is missing columns: {sorted(missing)}")

    targets = targets.sort_values("rank").head(TARGET_TOP_N).copy()

    win = crop_window()
    with rasterio.open(CANTON_ID_TIF) as src:
        canton = src.read(1, window=win)
    in_canton = np.isin(canton, TARGET_CANTON_IDS)

    if in_canton.shape != release.shape:
        raise ValueError(
            f"Canton crop shape {in_canton.shape} != release shape {release.shape}"
        )

    injectable = None
    if MICP_INJECTION_TIF and os.path.exists(MICP_INJECTION_TIF):
        with rasterio.open(MICP_INJECTION_TIF) as src:
            injection = src.read(1, window=win)
        if injection.shape != release.shape:
            raise ValueError(
                f"Injection crop shape {injection.shape} != release shape {release.shape}"
            )
        injectable = injection == MICP_INJECTABLE_VALUE
    else:
        print(
            f"[warning] injectability raster not found: {MICP_INJECTION_TIF}\n"
            "          target/source linkage will still run, but treatability "
            "fractions will be unavailable."
        )

    return (
        release,
        dem,
        target_ids,
        targets,
        in_canton,
        injectable,
        profile,
        transform,
    )


def cluster_release(release, in_canton):
    """Label all 8-neighbour connected baseline release clusters in the canton."""
    mask = release & in_canton
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    cluster_table = pd.DataFrame(
        {
            "cluster_id": np.arange(1, n + 1, dtype=int),
            "cells": sizes.astype(int),
        }
    )
    cluster_table["area_m2"] = cluster_table["cells"] * CELL_AREA_M2
    cluster_table["area_ha"] = cluster_table["area_m2"] / 10_000.0
    return lab, cluster_table


def write_binary_tif(path, mask, profile):
    p = profile.copy()
    p.update(dtype="uint8", nodata=0, count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(mask.astype(np.uint8), 1)


def _abs_project_path(path):
    return os.path.abspath(path)


def find_flowpy_zdelta(workdir):
    """Find newest backcalculation.tif in a requested workdir or Flow-Py sibling result."""
    root = Path(workdir)

    search_roots = []
    if root.exists():
        search_roots.append(root)

    parent = root.parent
    if parent.exists():
        search_roots.extend(p for p in parent.glob(f"{root.name}res_*") if p.is_dir())

    matches = []
    for r in search_roots:
        matches.extend(r.rglob("backcalculation.tif"))

    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def target_infra_path(rank):
    return f"{INFRA_DIR}/target_{int(rank):03d}.tif"


def target_workdir(rank):
    return f"{BACKTRACE_ROOT}/target_{int(rank):03d}_backtrace"


def ensure_target_infra(rank, target_ids, profile):
    mask = target_ids == int(rank)
    if not mask.any():
        raise ValueError(f"Target rank #{rank} does not occur in {TARGET_ID_TIF}")

    path = target_infra_path(rank)
    write_binary_tif(path, mask, profile)
    return path, mask


def run_or_reuse_target_backtrace(rank, infra_tif):
    """Run one infra back-trace so target identity is preserved."""
    workdir = target_workdir(rank)
    existing = find_flowpy_zdelta(workdir)

    if (
        existing
        and REUSE_EXISTING_TARGET_BACKTRACES
        and not FORCE_RERUN_TARGET_BACKTRACES
    ):
        print(f"  target #{rank}: reusing {existing}")
        return existing

    if not AUTO_RUN_TARGET_BACKTRACES:
        print(f"  target #{rank}: no existing per-target back-trace")
        return None

    flowpy_code = _abs_project_path(FLOWPY_CODE_DIR)
    flowpy_main = os.path.join(flowpy_code, FLOWPY_MAIN)
    if not os.path.exists(flowpy_main):
        raise FileNotFoundError(
            f"Flow-Py main script not found: {flowpy_main}\n" "Adjust FLOWPY_CODE_DIR."
        )

    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("Could not find conda on PATH.")

    dem = _abs_project_path(f"{INPUT_DIR}/dem.tif")
    release = _abs_project_path(f"{INPUT_DIR}/release.tif")
    infra = _abs_project_path(infra_tif)
    workdir_abs = _abs_project_path(workdir)
    os.makedirs(workdir_abs, exist_ok=True)

    cmd = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        FLOWPY_CONDA_ENV,
        "python",
        FLOWPY_MAIN,
        str(FLOWPY_ALPHA),
        str(FLOWPY_EXP),
        workdir_abs,
        dem,
        release,
        f"infra={infra}",
        *FLOWPY_EXTRA_ARGS,
    ]

    print(f"\n  target #{rank}: running individual Flow-Py infra back-trace")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=flowpy_code, check=True)

    zdelta = find_flowpy_zdelta(workdir)
    if zdelta is None:
        raise FileNotFoundError(
            f"Flow-Py finished for target #{rank}, but no z_delta.tif was found."
        )

    return zdelta


def read_backtrace_mask(path, expected_shape):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Back-trace raster {path} shape {arr.shape} != expected {expected_shape}"
        )
    return np.isfinite(arr) & (arr > 0)


def hillshade(dem):
    finite = np.isfinite(dem)
    filled = np.where(finite, dem, np.nanmin(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(filled, vert_exag=1.5, dx=CELL_SIZE_M, dy=CELL_SIZE_M)


def analyse_one_target(
    target_row,
    target_mask,
    backtrace_mask,
    release,
    lab,
    cluster_table,
    injectable,
    in_canton,
):
    """Return all connected release cells and their injectable subset."""
    rank = int(target_row["rank"])
    source_cells = backtrace_mask & release & in_canton
    source_cluster_ids = [
        int(cid) for cid in np.unique(lab[source_cells]) if int(cid) > 0
    ]

    if injectable is not None:
        injectable_source = source_cells & injectable
        noninjectable_source = source_cells & ~injectable
    else:
        injectable_source = np.zeros(release.shape, dtype=bool)
        noninjectable_source = source_cells.copy()

    links = []
    ctab = cluster_table.set_index("cluster_id") if not cluster_table.empty else None
    for cid in source_cluster_ids:
        connected = source_cells & (lab == cid)
        inj = injectable_source & (lab == cid)
        n_connected = int(connected.sum())
        n_inj = int(inj.sum()) if injectable is not None else np.nan
        links.append(
            {
                "target_rank": rank,
                "target_asset_type": target_row["asset_type"],
                "target_asset": target_row["asset"],
                "target_risk_chf_per_year": float(target_row["risk_chf_per_year"]),
                "release_cluster_id": cid,
                "cluster_total_cells": int(ctab.loc[cid, "cells"]),
                "cluster_total_area_ha": float(ctab.loc[cid, "area_ha"]),
                "connected_release_cells": n_connected,
                "connected_release_area_ha": n_connected * CELL_AREA_M2 / 10_000.0,
                "connected_injectable_cells": n_inj,
                "connected_injectable_area_ha": (
                    n_inj * CELL_AREA_M2 / 10_000.0 if np.isfinite(n_inj) else np.nan
                ),
            }
        )

    n_source = int(source_cells.sum())
    n_inj = int(injectable_source.sum()) if injectable is not None else np.nan
    frac = n_inj / n_source if injectable is not None and n_source > 0 else np.nan
    summary = {
        "target_rank": rank,
        "asset_type": target_row["asset_type"],
        "asset": target_row["asset"],
        "target_risk_chf_per_year": float(target_row["risk_chf_per_year"]),
        "target_x_lv95": float(target_row["x_lv95"]),
        "target_y_lv95": float(target_row["y_lv95"]),
        "n_contributing_release_clusters": len(source_cluster_ids),
        "connected_release_cells": n_source,
        "connected_release_area_ha": n_source * CELL_AREA_M2 / 10_000.0,
        "injectable_connected_release_cells": n_inj,
        "injectable_connected_release_area_ha": (
            n_inj * CELL_AREA_M2 / 10_000.0 if np.isfinite(n_inj) else np.nan
        ),
        "injectable_source_fraction": frac,
    }
    return summary, links, source_cells, injectable_source, noninjectable_source


def _show_mask(ax, mask, color, alpha):
    if mask is not None and mask.any():
        ax.imshow(
            np.ma.masked_where(~mask, np.ones(mask.shape, dtype=np.uint8)),
            cmap=ListedColormap([color]),
            alpha=alpha,
            interpolation="nearest",
        )


def plot_overall_target_map(
    dem,
    in_canton,
    release,
    target_ids,
    targets,
    source_union,
    injectable_union,
    outpath,
):
    shade = hillshade(dem)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(shade, cmap="gray", vmin=0, vmax=2.0)
    ax.contour(
        in_canton.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.1,
        alpha=0.85,
    )

    baseline = release & in_canton
    other_release = baseline & ~source_union
    connected_noninj = source_union & ~injectable_union
    _show_mask(ax, other_release, "#bdbdbd", 0.20)
    _show_mask(ax, connected_noninj, "#f39c12", 0.85)
    _show_mask(ax, injectable_union, "#8a6bd1", 0.95)

    selected_ranks = set(targets["rank"].astype(int))
    all_target_mask = np.isin(target_ids, list(selected_ranks))
    _show_mask(ax, all_target_mask, "#1565c0", 1.0)

    for _, r in targets.iterrows():
        rr, cc = rasterio.transform.rowcol(
            transform_global, float(r["x_lv95"]), float(r["y_lv95"])
        )
        ax.text(
            cc + 3,
            rr - 3,
            f"#{int(r['rank'])}",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.10", facecolor="white", alpha=0.88, linewidth=0.4
            ),
            zorder=10,
        )

    ax.legend(
        handles=[
            Patch(facecolor="#1565c0", label="highest-damage target sites"),
            Patch(
                facecolor="#f39c12",
                label="release cells, not MICP-injectable",
            ),
            Patch(facecolor="#8a6bd1", label="release cells, MICP-injectable"),
            Patch(
                facecolor="#bdbdbd", alpha=0.20, label="other baseline release cells"
            ),
        ],
        loc="upper right",
        framealpha=0.92,
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Target-connected release sources and MICP injectability, canton {CANTON_LABEL}\n"
        f"top {len(targets)} highest-risk damage sites",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_one_target_map(
    target_row,
    dem,
    in_canton,
    release,
    target_mask,
    source_cells,
    injectable_source,
    summary,
    outpath,
):
    shade = hillshade(dem)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(shade, cmap="gray", vmin=0, vmax=2.0)
    ax.contour(
        in_canton.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.0,
        alpha=0.85,
    )

    baseline = release & in_canton
    other_release = baseline & ~source_cells
    noninjectable_source = source_cells & ~injectable_source
    _show_mask(ax, other_release, "#bdbdbd", 0.16)
    _show_mask(ax, noninjectable_source, "#f39c12", 0.90)
    _show_mask(ax, injectable_source, "#8a6bd1", 0.98)
    _show_mask(ax, target_mask, "#1565c0", 1.0)

    frac = summary["injectable_source_fraction"]
    frac_txt = f"{100*frac:.1f}%" if pd.notna(frac) else "n/a"
    ax.legend(
        handles=[
            Patch(facecolor="#1565c0", label="damage target"),
            Patch(facecolor="#f39c12", label="Not MICP-injectable"),
            Patch(facecolor="#8a6bd1", label="MICP-injectable"),
            Patch(
                facecolor="#bdbdbd", alpha=0.16, label="other baseline release cells"
            ),
        ],
        loc="upper right",
        framealpha=0.92,
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Target #{int(target_row['rank'])}: {target_row['asset_type']} — CHF {target_row['risk_chf_per_year']:,.0f}/yr\n"
        f"{summary['n_contributing_release_clusters']} contributing release clusters | "
        f"{frac_txt} of connected release area MICP-injectable",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_target_injectability(summary_df, outpath):
    if summary_df.empty:
        return
    d = (
        summary_df.sort_values("target_risk_chf_per_year", ascending=False)
        .copy()
        .iloc[::-1]
    )
    labels = [
        f"#{int(r.target_rank)} {r.asset_type} | CHF {r.target_risk_chf_per_year:,.0f}/yr"
        for r in d.itertuples()
    ]
    vals = 100.0 * d["injectable_source_fraction"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 0.48 * len(d) + 2.5))
    ax.barh(labels, vals)
    for i, (_, r) in enumerate(d.iterrows()):
        if pd.notna(r["injectable_source_fraction"]):
            txt = f" {100*r['injectable_source_fraction']:.1f}% | {r['injectable_connected_release_area_ha']:.1f} of {r['connected_release_area_ha']:.1f} ha"
        else:
            txt = " injectability unavailable"
        ax.text(min(100.0, float(vals.iloc[i])) + 1.0, i, txt, va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("MICP-injectable share of target-connected release area [%]")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        "How much of the release area feeding each high-damage target is MICP-injectable?",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def main():
    global transform_global
    release, dem, target_ids, targets, in_canton, injectable, profile, transform = (
        load_inputs()
    )
    transform_global = transform

    print("=" * 76)
    print(f"SIMPLE TARGET-CENTRIC MICP SCREENING — canton {CANTON_LABEL}")
    print("=" * 76)
    print(f"targets analysed: top {len(targets)}")
    print("Metric: MICP-injectable connected release area / all connected release area")
    print("No practical-patch filter and no avoided-EAD claim.")

    lab, cluster_table = cluster_release(release, in_canton)
    summary_rows, link_rows = [], []
    source_union = np.zeros(release.shape, dtype=bool)
    injectable_union = np.zeros(release.shape, dtype=bool)

    for _, target_row in targets.iterrows():
        rank = int(target_row["rank"])
        print(
            f"\nTARGET #{rank}: {target_row['asset_type']} — CHF {target_row['risk_chf_per_year']:,.0f}/yr"
        )
        infra_tif, target_mask = ensure_target_infra(rank, target_ids, profile)
        try:
            zdelta = run_or_reuse_target_backtrace(rank, infra_tif)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"  [warning] target #{rank} back-trace failed: {exc}")
            continue
        if not zdelta:
            print(f"  [warning] target #{rank} has no per-target back-trace result")
            continue

        backtrace_mask = read_backtrace_mask(zdelta, release.shape)
        summary, links, source_cells, injectable_source, _ = analyse_one_target(
            target_row,
            target_mask,
            backtrace_mask,
            release,
            lab,
            cluster_table,
            injectable,
            in_canton,
        )
        summary["backtrace_zdelta"] = zdelta
        summary_rows.append(summary)
        link_rows.extend(links)
        source_union |= source_cells
        injectable_union |= injectable_source

        frac = summary["injectable_source_fraction"]
        frac_txt = f"{100*frac:.1f}%" if pd.notna(frac) else "n/a"
        print(
            f"  contributing release clusters : {summary['n_contributing_release_clusters']}"
        )
        print(
            f"  connected release area        : {summary['connected_release_area_ha']:.2f} ha"
        )
        print(
            f"  MICP-injectable connected area: {summary['injectable_connected_release_area_ha']:.2f} ha"
        )
        print(f"  injectable share              : {frac_txt}")

        if rank <= TARGET_TOP_N:
            plot_one_target_map(
                target_row,
                dem,
                in_canton,
                release,
                target_mask,
                source_cells,
                injectable_source,
                summary,
                f"{PER_TARGET_FIG_DIR}/target_{rank:03d}.png",
            )

    summary_df = pd.DataFrame(summary_rows)
    links_df = pd.DataFrame(link_rows)
    summary_csv = f"{OUTDIR}/target_micp_injectability_summary.csv"
    links_csv = f"{OUTDIR}/target_release_links.csv"
    summary_df.to_csv(summary_csv, index=False)
    links_df.to_csv(links_csv, index=False)
    print(f"\n-> {summary_csv}\n-> {links_csv}")

    if not summary_df.empty:
        plotted_targets = targets[
            targets["rank"].isin(summary_df["target_rank"])
        ].copy()
        plot_overall_target_map(
            dem,
            in_canton,
            release,
            target_ids,
            plotted_targets,
            source_union,
            injectable_union,
            f"{FIG_DIR}/target_connected_release_injectability_map.png",
        )
        plot_target_injectability(
            summary_df, f"{FIG_DIR}/highest_damage_target_injectability.png"
        )

    report = [
        "SIMPLE TARGET-CENTRIC MICP SCREENING",
        "=" * 76,
        "Each high-risk target is back-traced separately so target identity is preserved.",
        "Connected source = baseline FoS<1 release cells intersecting that target's Flow-Py backtrace.",
        "MICP-injectable source = connected source cells where the injection raster equals the configured injectable value.",
        "Injectable share = MICP-injectable connected release area / total connected release area.",
        "No practical treatment-patch filter is applied.",
        "This is a screening metric, not avoided EAD; avoided EAD requires an FoS_MICP scenario and new forward Flow-Py run.",
        "",
        f"Targets requested: top {TARGET_TOP_N}",
        f"Targets successfully analysed: {len(summary_df)}",
        "",
    ]
    if not summary_df.empty:
        report.append("TARGET RESULTS")
        for _, r in summary_df.sort_values(
            "target_risk_chf_per_year", ascending=False
        ).iterrows():
            pct = r["injectable_source_fraction"]
            pct_txt = f"{100*pct:.1f}%" if pd.notna(pct) else "n/a"
            report.append(
                f"#{int(r['target_rank'])} {r['asset_type']}: CHF {r['target_risk_chf_per_year']:,.0f}/yr | "
                f"{int(r['n_contributing_release_clusters'])} source clusters | "
                f"{r['connected_release_area_ha']:.2f} ha connected | "
                f"{r['injectable_connected_release_area_ha']:.2f} ha injectable | {pct_txt} injectable"
            )
    report_path = f"{OUTDIR}/target_micp_injectability_report.txt"
    Path(report_path).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"-> {report_path}\n\nDone.")


if __name__ == "__main__":
    main()
