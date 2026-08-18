"""
02_evaluate_local_slopes.py — AUC with REAL slope angles vs the fixed 33 deg.

Instead of const.BETA_DEG for every point, it samples the local slope from the
25 m raster at each landslide/control location, two ways:
  - single  : the exact pixel at the coordinate
  - max3x3  : the steepest pixel in a 3x3 window (robust to the ~1-pixel coordinate
              uncertainty of the inventory, which often marks the deposit, not the scar)

Three variants are compared on the SAME point set: fixed 33, single, max3x3.
Only points whose sampled angle is inside the infinite-slope validity band
[15, 45] deg contribute for the real-angle variants.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from core import data_loader as dl
from core import constants as const
from core import val_metrics as vm
from pipeline.utils import constants as auct
from pipeline.utils import utils as autils

_SLOPE_CACHE = {}
OUTPUT_DIR = "output/spatial/02_local_slope_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_LABEL = f"Fixed {const.BETA_DEG}°"


def get_slopes(src, x, y):
    """Return (single_pixel_deg, max_3x3_deg) at (easting, northing). NaN if off-raster."""
    key = (round(x), round(y))
    if key in _SLOPE_CACHE:
        return _SLOPE_CACHE[key]
    try:
        row, col = src.index(x, y)
        a = src.read(
            1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=np.nan
        ).astype("float32")
    except Exception:
        _SLOPE_CACHE[key] = (np.nan, np.nan)
        return _SLOPE_CACHE[key]
    if src.nodata is not None:
        a[a == src.nodata] = np.nan
    single = float(a[1, 1])
    mx = float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
    res = (single, mx)
    _SLOPE_CACHE[key] = res
    return res


def main():
    inv = dl.load_wsl_usable_inventory()

    variant_names = ["Single Pixel", "Max 3x3", FIXED_LABEL]

    res = {
        name: {"hydro_pos": [], "hydro_neg": [], "full_pos": [], "full_neg": []}
        for name in variant_names
    }

    regime_counts = {
        name: {"hydro": 0, "dry": 0, "invalid": 0} for name in variant_names
    }

    with rasterio.open(auct.SLOPE_TIF) as src:
        for i, ev in inv.iterrows():
            rx, ry = dl.to_lv95(ev["x"], ev["y"])
            single, max3 = get_slopes(src, rx, ry)

            # Raw angles
            raw_angles = {
                "Single Pixel": single,
                "Max 3x3": max3,
                FIXED_LABEL: const.BETA_DEG,
            }

            regimes = {name: autils.classify_slope(a) for name, a in raw_angles.items()}
            for name, regime in regimes.items():
                regime_counts[name][regime] += 1

            controls = vm.find_control_date(
                ev["x"], ev["y"], ev["date"], inv, n=auct.CONTROLS_PER_EVENT
            )

            # --- positive (event) ---
            for name, beta in raw_angles.items():
                regime = regimes[name]
                if regime == "invalid":
                    continue

                f = autils.min_fos_at(rx, ry, ev["date"], beta_deg=beta)
                if not np.isfinite(f):
                    continue

                res[name]["full_pos"].append(f)
                if regime == "hydro":
                    res[name]["hydro_pos"].append(f)

                # --- negatives (matched controls) ---
                for control_date in controls:
                    g = autils.min_fos_at(rx, ry, control_date, beta_deg=beta)
                    if np.isfinite(g):
                        res[name]["full_neg"].append(g)
                        if regime == "hydro":
                            res[name]["hydro_neg"].append(g)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(inv)} processed")

    # --- regime counts ---
    print(f"\n{'variant':>14}  {'hydro':>7} {'dry':>6} {'invalid':>8}")
    for name in variant_names:
        c = regime_counts[name]
        print(f"{name:>14}  {c['hydro']:7d} {c['dry']:6d} {c['invalid']:8d}")

    # --- AUC table ---
    print(
        f"\n{'variant':>14} {'AUC(hydro)':>11} {'AUC(full)':>10} "
        f"{'J(full)':>8} {'MCC(full)':>10}"
    )

    results = {}
    for name in variant_names:
        d = res[name]
        hp, hn = np.array(d["hydro_pos"]), np.array(d["hydro_neg"])
        fp, fn = np.array(d["full_pos"]), np.array(d["full_neg"])
        if len(fp) == 0 or len(fn) == 0:
            print(f"{name:>14}  insufficient data")
            continue

        auc_hydro = autils.auc_score(hp, hn) if len(hp) and len(hn) else np.nan
        auc_full = autils.auc_score(fp, fn)
        j = vm.youden_j(fp <= 1.0, fn <= 1.0)
        m = vm.mcc(fp <= 1.0, fn <= 1.0)
        results[name] = {
            "auc_hydro": auc_hydro,
            "auc_full": auc_full,
            "j_full": j,
            "mcc_full": m,
            "full_pos": fp,
            "full_neg": fn,
        }
        auc_hydro_str = (
            f"{auc_hydro:11.3f}" if np.isfinite(auc_hydro) else f"{'n/a':>11}"
        )
        print(f"{name:>14} {auc_hydro_str} {auc_full:10.3f} {j:8.3f} {m:10.3f}")

    # Save Results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "slope_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,auc_hydro,auc_full,j_full,mcc_full\n")
        for name, metrics in results.items():
            f.write(
                f"{name},{metrics['auc_hydro']},{metrics['auc_full']},"
                f"{metrics['j_full']},{metrics['mcc_full']}\n"
            )
    print(f"\nResults saved to {csv_path}")

    # --- plot 1: ROC overlay ---
    colors = {FIXED_LABEL: "gray", "Single Pixel": "steelblue", "Max 3x3": "firebrick"}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, r in results.items():
        fpr, tpr = autils.roc(r["full_pos"], r["full_neg"])
        ax.plot(
            fpr, tpr, color=colors[name], lw=2, label=f"{name}: AUC={r['auc_full']:.3f}"
        )
    ax.plot([0, 1], [0, 1], color="black", ls="--", alpha=0.4, label="random (0.50)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_slope_compare.png", dpi=150)
    plt.close(fig)

    if "Max 3x3" in results and FIXED_LABEL in results:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        pos3, neg3 = results["Max 3x3"]["full_pos"], results["Max 3x3"]["full_neg"]
        bins = np.linspace(0, 4, 41)
        a1.hist(
            neg3,
            bins=bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="controls",
        )
        a1.hist(
            pos3, bins=bins, density=True, alpha=0.5, color="firebrick", label="events"
        )
        a1.axvline(1.0, color="black", ls="--", alpha=0.6, label="FoS = 1")
        a1.set_xlabel("min FoS in window")
        a1.set_ylabel("density")
        a1.set_title(f"real angle (max 3x3): AUC={results['Max 3x3']['auc_full']:.3f}")
        a1.legend()

        neg_fixed = results[FIXED_LABEL]["full_neg"]
        a2.hist(
            neg_fixed,
            bins=bins,
            density=True,
            alpha=0.5,
            color="gray",
            label=FIXED_LABEL,
        )
        a2.hist(
            neg3,
            bins=bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="real angle (max 3x3)",
        )
        a2.axvline(1.0, color="black", ls="--", alpha=0.6)
        a2.set_xlabel("min FoS in window (controls only)")
        a2.set_ylabel("density")
        a2.set_title("controls: fixed vs real angle")
        a2.legend()

        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/fos_slope_fanout.png", dpi=150)
        plt.close(fig)

    print(
        f"\n-> {OUTPUT_DIR}/roc_slope_compare.png | {OUTPUT_DIR}/fos_slope_fanout.png"
    )

    # Plot J and MCC vs slope angle for the real-angle variants
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    for name in ["Single Pixel", "Max 3x3"]:
        d = res[name]
        fp, fn = np.array(d["full_pos"]), np.array(d["full_neg"])
        j = vm.youden_j(fp <= 1.0, fn <= 1.0)
        m = vm.mcc(fp <= 1.0, fn <= 1.0)
        ax1.plot(
            [name],
            [j],
            "o",
            color="purple" if name == "Single Pixel" else "firebrick",
            label=f"{name}: J={j:.3f}",
        )
        ax1.plot(
            [name],
            [m],
            "s",
            color="darkorange" if name == "Single Pixel" else "steelblue",
            label=f"{name}: MCC={m:.3f}",
        )
    ax1.set_xlabel("Variant")
    ax1.set_ylabel("Score")
    ax1.set_title("Metrics vs slope angle variant")
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/threshold_metrics_vs_slope_variant.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
