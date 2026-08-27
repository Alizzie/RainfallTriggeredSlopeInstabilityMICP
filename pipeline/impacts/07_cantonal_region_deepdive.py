"""
07_cantonal_region_deepdive.py - Canton version of region_deepdive_v2.py.

Output (per canton, output/impacts/06_cantonal/canton_<id>/):
    roc_canton_<id>_fixed_vs_real.png
    fos_distribution_canton_<id>_fixed.png
    fos_distribution_canton_<id>_real.png
Plus: canton_deepdive_comparison.csv and a log file.
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

TARGET_CANTONS = [
    1,
    15,
    22,
]
CANTON_ID_TIF = "data/swissboundaries/canton_id_25m.tif"
CANTON_NAMES_CSV = "data/swissboundaries/canton_names.csv"

CONTROLS_PER_EVENT = auct.CONTROLS_PER_EVENT
OUTDIR = "output/impacts/07_cantonal_deepdive"


def load_canton_names():
    if os.path.exists(CANTON_NAMES_CSV):
        names_df = pd.read_csv(CANTON_NAMES_CSV)
        return dict(zip(names_df["KANTONSNUM"], names_df["NAME"]))
    return {}


def load_canton_raster():
    with rasterio.open(CANTON_ID_TIF) as src:
        canton_id = src.read(1).astype(np.int16)
        transform = src.transform
    return canton_id, transform


def canton_at(x, y, canton_id, transform):
    """Point-sample the canton ID raster - no vector lookup needed."""
    row, col = rasterio.transform.rowcol(transform, x, y)
    if row < 0 or row >= canton_id.shape[0] or col < 0 or col >= canton_id.shape[1]:
        return None
    val = int(canton_id[row, col])
    return val if val > 0 else None


def load_slope_raster():
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta_raster = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    return beta_raster, transform, nodata


def real_beta_at(x, y, beta_raster, transform, nodata):
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


def events_in_canton(inv, canton_target, canton_id, transform):
    assigned = inv.apply(
        lambda r: canton_at(r["x"], r["y"], canton_id, transform), axis=1
    )
    return inv[assigned == canton_target].reset_index(drop=True)


def score_method(canton_events, inv, beta_lookup, log, method_name):
    pos_scores, neg_scores = [], []
    event_flags, control_flags = [], []
    n_no_data, n_no_beta, n_no_control = 0, 0, 0

    for _, ev in canton_events.iterrows():
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


def run_canton(
    cid,
    canton_name,
    inv,
    canton_id_raster,
    canton_transform,
    beta_raster,
    beta_transform,
    beta_nodata,
    log,
):
    canton_events = events_in_canton(inv, cid, canton_id_raster, canton_transform)
    label = f"{cid} ({canton_name})" if canton_name else str(cid)
    log(f"\n{'=' * 66}")
    log(f"CANTON {label}")
    log(f"{'=' * 66}")
    log(f"Events physically inside canton {label}: {len(canton_events)}")

    if len(canton_events) < 5:
        log(f"  Too few events (<5) for a meaningful AUC - skipping canton.")
        return []

    canton_dir = f"{OUTDIR}/canton_{cid}"
    os.makedirs(canton_dir, exist_ok=True)

    log(f"\n-- fixed ({const.BETA_DEG} deg) --")
    fixed = score_method(canton_events, inv, lambda ev: const.BETA_DEG, log, "fixed")

    log(f"\n-- real (per-pixel slope raster) --")
    real = score_method(
        canton_events,
        inv,
        lambda ev: real_beta_at(
            ev["x"], ev["y"], beta_raster, beta_transform, beta_nodata
        ),
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
            pos, neg, auc, f"{canton_dir}/fos_distribution_canton_{cid}_{name}.png"
        )

        rows.append(
            {
                "canton_id": cid,
                "canton_name": canton_name,
                "method": name,
                "n_events_total": len(canton_events),
                "n_events_usable": len(pos),
                "n_controls": len(neg),
                "auc": round(auc, 3),
                "youden_j": round(j, 3),
                "mcc": round(m, 3),
                "tpr": round(tpr, 3),
                "fpr": round(fpr, 3),
            }
        )

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
        ax.set_title(f"Canton {label}: Fixed 30° vs Real Slope Angle")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = f"{canton_dir}/roc_canton_{cid}_fixed_vs_real.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        log(f"\n  -> {roc_path}")

    return rows


def main():
    inv = dl.load_wsl_usable_inventory()
    os.makedirs(OUTDIR, exist_ok=True)

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log(f"Canton deep-dive (fixed vs real slope angle): {TARGET_CANTONS}")
    log(f"National inventory: {len(inv)} events")

    canton_id_raster, canton_transform = load_canton_raster()
    canton_names = load_canton_names()
    beta_raster, beta_transform, beta_nodata = load_slope_raster()

    all_rows = []
    for cid in TARGET_CANTONS:
        name = canton_names.get(cid, "")
        all_rows.extend(
            run_canton(
                cid,
                name,
                inv,
                canton_id_raster,
                canton_transform,
                beta_raster,
                beta_transform,
                beta_nodata,
                log,
            )
        )

    if all_rows:
        comp_df = pd.DataFrame(all_rows)
        comp_path = f"{OUTDIR}/canton_deepdive_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        log(f"\n{'=' * 66}")
        log("FULL COMPARISON (fixed vs real, per canton)")
        log(f"{'=' * 66}")
        log(comp_df.to_string(index=False))
        log(f"\n-> {comp_path}")

    with open(f"{OUTDIR}/canton_deepdive_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\n-> {OUTDIR}/canton_deepdive_log.txt")


if __name__ == "__main__":
    main()
