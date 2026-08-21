"""
region_deepdive.py - Region-specific temporal validation for target regions
(default: 47 and 38 - flagged as closer to Storme's soil properties), computing
AUC / Youden's J / MCC TWICE per region:

    "fixed" : const.BETA_DEG (30 deg), the national Phase-1 winner
    "real"  : the actual per-pixel slope angle at each event/control location,
              sampled from the slope raster (auct.SLOPE_TIF)

Output (per region, output/impacts/03_regional/region_<id>/):
    roc_region_<id>_fixed_vs_real.png   - both ROC curves on one plot
    fos_distribution_region_<id>_fixed.png
    fos_distribution_region_<id>_real.png
Plus: region_deepdive_comparison.csv (one row per region x method) and a log file.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import data_loader as dl
from pipeline.utils import utils as ut
from core import constants as const
from pipeline.utils import constants as auct
from core import val_metrics as vm

TARGET_REGIONS = [47, 38]
CONTROLS_PER_EVENT = auct.CONTROLS_PER_EVENT
OUTDIR = "output/impacts/05_regional_38_47"


def load_slope_raster():
    """Load the national slope raster once; reused for every region/event."""
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta_raster = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    return beta_raster, transform, nodata


def real_beta_at(x, y, beta_raster, transform, nodata):
    """Real per-pixel slope angle [deg] at (x, y), or NaN if outside the
    valid 15-45 deg soil mask / raster extent / nodata."""
    row, col = rasterio.transform.rowcol(transform, x, y)
    if row < 0 or row >= beta_raster.shape[0] or col < 0 or col >= beta_raster.shape[1]:
        return np.nan
    val = beta_raster[row, col]
    if nodata is not None and val == nodata:
        return np.nan
    if not np.isfinite(val):
        return np.nan
    if val < auct.BETA_MIN or val > auct.BETA_MAX:
        return np.nan
    return float(val)


# ------------------------------------------------------------------- region filter


def events_in_region(inv, region_id, max_snap_m=2000.0):
    """Filter the inventory to events physically inside one region."""
    assigned = inv.apply(
        lambda r: dl.assign_region(r["x"], r["y"], max_snap_m=max_snap_m), axis=1
    )
    return inv[assigned == region_id].reset_index(drop=True)


# ------------------------------------------------------------------- one method


def score_method(region_events, inv, beta_lookup, log, method_name):
    """
    Compute event/control min-FoS arrays for ONE beta strategy.

    beta_lookup(ev) -> beta_deg for this event's location, or None to use the
    module-level fixed angle. Returns (pos, neg, event_flags, control_flags,
    n_no_data, n_no_beta, n_no_control) or None if too few usable events.
    """
    pos_scores, neg_scores = [], []
    event_flags, control_flags = [], []
    n_no_data, n_no_beta, n_no_control = 0, 0, 0

    for _, ev in region_events.iterrows():
        beta_deg = beta_lookup(ev)
        if beta_deg is None or (isinstance(beta_deg, float) and np.isnan(beta_deg)):
            n_no_beta += 1
            continue

        event_fos = ut.min_fos_at(ev["x"], ev["y"], ev["date"], beta_deg=beta_deg)
        if np.isnan(event_fos):
            n_no_data += 1
            continue

        controls = vm.find_control_date(
            ev["x"], ev["y"], ev["date"], inv, n=CONTROLS_PER_EVENT
        )
        if not controls:
            n_no_control += 1
            continue

        control_fos_list = [
            ut.min_fos_at(ev["x"], ev["y"], cd, beta_deg=beta_deg) for cd in controls
        ]
        control_fos_list = [c for c in control_fos_list if not np.isnan(c)]
        if not control_fos_list:
            n_no_control += 1
            continue

        pos_scores.append(event_fos)
        neg_scores.extend(control_fos_list)
        event_flags.append(event_fos <= 1.0)
        control_flags.extend([c <= 1.0 for c in control_fos_list])

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    log(
        f"  [{method_name}] usable events: {len(pos_scores)}  "
        f"(dropped {n_no_beta} no_valid_slope, {n_no_data} no_rain_data, {n_no_control} no_control)"
    )
    log(f"  [{method_name}] matched controls: {len(neg_scores)}")

    if len(pos_scores) < 5 or len(neg_scores) < 5:
        log(f"  [{method_name}] too few usable events/controls - skipping.")
        return None

    return pos_scores, neg_scores, event_flags, control_flags


# ------------------------------------------------------------------- per region


def run_region(region_id, inv, beta_raster, transform, nodata, log):
    region_events = events_in_region(inv, region_id)
    log(f"\n{'=' * 66}")
    log(f"REGION {region_id}")
    log(f"{'=' * 66}")
    log(f"Events physically inside region {region_id}: {len(region_events)}")

    if len(region_events) < 5:
        log(f"  Too few events (<5) for a meaningful AUC - skipping region.")
        return []

    region_dir = f"{OUTDIR}/region_{region_id}"
    os.makedirs(region_dir, exist_ok=True)

    # --- method 1: fixed 30 deg (Phase 1's national winner) ---
    log(f"\n-- fixed ({const.BETA_DEG} deg) --")
    fixed = score_method(region_events, inv, lambda ev: const.BETA_DEG, log, "fixed")

    # --- method 2: real per-pixel slope angle at each location ---
    log(f"\n-- real (per-pixel slope raster) --")
    real = score_method(
        region_events,
        inv,
        lambda ev: real_beta_at(ev["x"], ev["y"], beta_raster, transform, nodata),
        log,
        "real",
    )

    methods = {"fixed": fixed, "real": real}
    rows = []
    roc_curves = {}

    for name, result in methods.items():
        if result is None:
            continue
        pos, neg, ev_flags, ctrl_flags = result

        auc = ut.auc_score(pos, neg)
        j = vm.youden_j(ev_flags, ctrl_flags)
        m = vm.mcc(ev_flags, ctrl_flags)
        tpr, fpr = vm.tpr_fpr(ev_flags, ctrl_flags)

        log(
            f"\n  [{name}] AUC={auc:.3f}  J={j:.3f}  MCC={m:.3f}  TPR={tpr:.1%}  FPR={fpr:.1%}"
        )

        fpr_curve, tpr_curve = ut.roc(pos, neg)
        roc_curves[name] = (fpr_curve, tpr_curve, auc)
        ut.plot_fos_distribution(
            pos,
            neg,
            auc,
            f"{region_dir}/fos_distribution_region_{region_id}_{name}.png",
        )

        rows.append(
            {
                "region_id": region_id,
                "method": name,
                "n_events_total": len(region_events),
                "n_events_usable": len(pos),
                "n_controls": len(neg),
                "auc": round(auc, 3),
                "youden_j": round(j, 3),
                "mcc": round(m, 3),
                "tpr": round(tpr, 3),
                "fpr": round(fpr, 3),
            }
        )

    # --- combined ROC: fixed vs real on one plot ---
    if roc_curves:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        colors = {"fixed": "firebrick", "real": "steelblue"}
        for name, (fpr_c, tpr_c, auc) in roc_curves.items():
            ax.plot(
                fpr_c,
                tpr_c,
                lw=2,
                color=colors.get(name),
                label=f"{name} (AUC={auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="random (0.50)")
        ax.set_xlabel("false positive rate (controls flagged)")
        ax.set_ylabel("true positive rate (events detected)")
        ax.set_title(f"Region {region_id}: Fixed 30° vs Real Slope Angle")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = f"{region_dir}/roc_region_{region_id}_fixed_vs_real.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        log(f"\n  -> {roc_path}")

    if "fixed" in [r["method"] for r in rows] and "real" in [r["method"] for r in rows]:
        f_auc = next(r["auc"] for r in rows if r["method"] == "fixed")
        r_auc = next(r["auc"] for r in rows if r["method"] == "real")
        winner = "real" if r_auc > f_auc else ("fixed" if f_auc > r_auc else "tie")
        log(
            f"\n  >> In region {region_id}: {winner} wins on AUC ({f_auc:.3f} fixed vs {r_auc:.3f} real)"
        )

    return rows


def main():
    inv = dl.load_wsl_usable_inventory()
    os.makedirs(OUTDIR, exist_ok=True)

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log(f"Region deep-dive (fixed vs real slope angle): {TARGET_REGIONS}")
    log(f"National inventory: {len(inv)} events")

    beta_raster, transform, nodata = load_slope_raster()

    all_rows = []
    for region_id in TARGET_REGIONS:
        all_rows.extend(run_region(region_id, inv, beta_raster, transform, nodata, log))

    if all_rows:
        comp_df = pd.DataFrame(all_rows)
        comp_path = f"{OUTDIR}/region_deepdive_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        log(f"\n{'=' * 66}")
        log("FULL COMPARISON (fixed vs real, per region)")
        log(f"{'=' * 66}")
        log(comp_df.to_string(index=False))
        log(f"\n-> {comp_path}")

    with open(f"{OUTDIR}/region_deepdive_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\n-> {OUTDIR}/region_deepdive_log.txt")


if __name__ == "__main__":
    main()
