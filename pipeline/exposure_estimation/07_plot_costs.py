"""
07_plot_costs.py - Turn the manually audited cost table into figures.

OUTPUT
cost_by_asset.png      the distributions, one row per asset
cost_convergence.png   is the sample big enough for a stable median?
cost_composition.png   how many figures, and how much money, per category
cost_summary.csv       the numbers, ready to paste into 08_risk_map.py
cost_summary_by_subtype.csv  conservative detail-based subcategories
cost_subtypes_<asset>.png    one subtype distribution figure per asset
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================== CONFIG =====

COSTS_CSV = "output/costs/ref_costs.csv"
OUTDIR = "output/costs"
os.makedirs(OUTDIR, exist_ok=True)

# Column names in the reviewed file.
COL_ASSET = "Type"
COL_COST = " Costs "
COL_DETAIL = "Detail"

# One landslide damaging one road or one building. Above this the figures are
# whole-storm totals, cantonal programmes, or damage from another hazard
# described in the same entry.
CAP_CHF = 5_000_000

# Categories naming a single asset, so they can carry a per-asset median.
PER_ASSET = {
    "road": "road",
    "railway": "railway",
    "building": "building",
    "forest": "forest",
    "agricultural land": "agricultural land",
    "power supply": "power supply",
}

# Real damage, but no single asset named - pooled figure only.
POOLED_ONLY = [
    "damage cost",
    "forest + agricultural land",
    "road + railway",
    "road + land",
    "road + water",
]

# Not landslide damage at all.
EXCLUDED = {
    "documentation": "cost of studying the hazard, not of damage",
    "protection work": "prevention, not damage",
    "emergency response": "deployment and clearing, not asset damage",
    "water": "river engineering - a different hazard",
}

COLORS = {
    "road": "#d84315",
    "building": "#c62828",
    "railway": "#4a148c",
    "forest": "#2e7d32",
    "agricultural land": "#f9a825",
    "power supply": "#0288d1",
}

# =========================================================== END CONFIG ====


def load():
    """Read the reviewed table and clean the two columns that matter."""
    df = pd.read_csv(COSTS_CSV, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(
        columns={
            COL_ASSET.strip(): "asset",
            COL_COST.strip(): "cost_chf",
            COL_DETAIL.strip(): "detail",
        }
    )

    df["asset"] = df["asset"].astype(str).str.strip().str.lower()
    df["cost_chf"] = pd.to_numeric(
        df["cost_chf"]
        .astype(str)
        .str.replace("CHF", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    return df.dropna(subset=["cost_chf"]).query("cost_chf > 0").reset_index(drop=True)


# ---------------------------------------------------------- sub-categories


def classify_subtype(row):
    """Assign a conservative detail-based subtype within each asset.

    The reviewed CSV is sparse in ``detail`` for some assets, so this deliberately
    returns ``unspecified`` unless the wording clearly supports a more specific
    label.  Asset remains the primary category; subtype is only a second level.
    """
    asset = str(row.get("asset", "")).strip().lower()
    d = str(row.get("detail", "") or "").strip().lower()

    # Common mojibake/ASCII variants found in the hand-reviewed CSV.
    for old, new in {
        "ã¼": "ü",
        "ã¤": "ä",
        "ã¶": "ö",
        "": "ü",
        "": "ä",
        "": "ö",
    }.items():
        d = d.replace(old, new)

    if not d or d == "nan":
        return "unspecified"

    def has(*terms):
        return any(t in d for t in terms)

    if asset == "road":
        if has("blockiert", "verschütt", "verschutt", "gesperrt", "sperr"):
            return "blocked / buried"
        if has("räum", "raum", "aufräum", "aufraum"):
            return "clearing"
        if has(
            "wiederherstell", "neubau", "neuer gehweg", "neubau brücke", "neubau bruck"
        ):
            return "restoration / rebuild"
        if has("sanier", "reparatur", "instand"):
            return "repair / sanierung"
        if has(
            "sicherung",
            "stützmauer",
            "stuetzmauer",
            "rammpf",
            "geogitter",
            "hangsicher",
            "schutzwand",
            "galerie",
        ):
            return "stabilisation / securing"
        if has("wanderweg", "schlittweg", "alpweg", "waldweg", "gehweg"):
            return "path / minor road"
        if has("brücke", "brucke", "brcke"):
            return "bridge"

    elif asset == "building":
        if has("stabilis"):
            return "stabilisation"
        if has("bedroh", "gefähr", "gefahr"):
            return "threatened building"
        if has("hausrat", "fahrhabe", "mobiliar"):
            return "contents / movable property"
        if has("grundstück", "grundstuck", "grundstck", "erschliess"):
            return "property / site"
        if has("keller", "wasser"):
            return "inundation / basement"

    elif asset == "railway":
        if has(
            "unterbruch",
            "einstellung",
            "bahnlose",
            "ersatzbus",
            "bahnersatz",
            "ertragsausfall",
            "einnahmeausfall",
        ):
            return "service interruption"
        if has("gefähr", "gefahr"):
            return "threatened infrastructure"
        if has("gleisverschieb", "reparatur", "sanier", "wiederherstell", "baukosten"):
            return "repair / restoration"

    elif asset == "forest":
        if has("wiederherstell", "aufforst"):
            return "forest restoration"
        if has("räum", "raum"):
            return "clearing"
        if has("waldzerst"):
            return "forest destruction"
        if has("sicherung"):
            return "terrain securing"

    elif asset == "agriculture land":
        if has("wiederherstell"):
            return "land restoration"

    elif asset == "power supply":
        if has("stromausfall", "ausfall"):
            return "power outage"
        if has("hochspannung", "mast", "leitung"):
            return "infrastructure damage"

    return "other described damage"


def add_subtypes(df):
    out = df.copy()
    out["subtype"] = out.apply(classify_subtype, axis=1)
    return out


# ------------------------------------------------------------------ plot 1


def plot_by_asset(per_asset, outpath):
    """One row per asset: every figure as a dot, with the quartile box."""
    assets = (
        per_asset.groupby("asset")["cost_chf"]
        .count()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(11, 0.75 * len(assets) + 2.5))
    rng = np.random.default_rng(0)

    for i, a in enumerate(assets):
        s = per_asset.loc[per_asset["asset"] == a, "cost_chf"]
        colour = COLORS.get(a, "#546e7a")

        # jittered dots, so overlapping values stay visible
        ax.scatter(
            s,
            i + rng.uniform(-0.16, 0.16, len(s)),
            s=22,
            color=colour,
            alpha=0.55,
            edgecolor="none",
            zorder=3,
        )

        p25, med, p75 = s.quantile([0.25, 0.5, 0.75])
        mean = s.mean()
        ax.plot(
            [p25, p75],
            [i, i],
            color=colour,
            lw=7,
            alpha=0.30,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot([med, med], [i - 0.28, i + 0.28], color="black", lw=2.2, zorder=4)
        ax.plot(
            [mean, mean], [i - 0.28, i + 0.28], color="red", lw=2.2, ls="--", zorder=5
        )

        # Keep the statistics in a fixed annotation column instead of placing
        # them after the largest dot.  This avoids labels jumping around or
        # colliding with extreme values on the log axis.
        ax.text(
            1.02,
            i,
            f"n={len(s)}  mean CHF {mean:,.0f}  median CHF {med:,.0f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )

    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)
    ax.set_xscale("log")
    ax.set_xlabel(
        f"reported cost per damaged asset [CHF, log scale, "
        f"capped at {CAP_CHF/1e6:.0f}M]"
    )
    ax.set_xlim(right=per_asset["cost_chf"].max() * 1.35)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        "What one damaged asset costs",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 0.74, 1))
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


def plot_subtypes_by_asset(per_asset, outdir):
    """Create one subtype distribution plot per asset.

    Only specifically classified rows are plotted. ``unspecified`` and the
    catch-all ``other described damage`` are omitted from the figure, but their
    counts are reported in the console so the classification coverage stays
    transparent.
    """
    specific = per_asset[
        ~per_asset["subtype"].isin(["unspecified", "other described damage"])
    ].copy()

    for asset in PER_ASSET:
        all_asset = per_asset[per_asset["asset"] == asset]
        g = specific[specific["asset"] == asset]
        if g.empty:
            print(
                f"  subtype plot skipped for {asset}: no specifically classified rows"
            )
            continue

        unclassified = int((all_asset["subtype"] == "unspecified").sum())
        generic = int((all_asset["subtype"] == "other described damage").sum())
        n_total = len(all_asset)
        n_specific = len(g)

        order = (
            g.groupby("subtype")["cost_chf"]
            .count()
            .sort_values(ascending=True)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.72 * len(order) + 2.4)))
        rng = np.random.default_rng(1)
        colour = COLORS.get(asset, "#546e7a")

        for i, subtype in enumerate(order):
            vals = g.loc[g["subtype"] == subtype, "cost_chf"]
            ax.scatter(
                vals,
                i + rng.uniform(-0.15, 0.15, len(vals)),
                s=24,
                color=colour,
                alpha=0.58,
                edgecolor="none",
                zorder=3,
            )

            p25, med, p75 = vals.quantile([0.25, 0.5, 0.75])
            ax.plot(
                [p25, p75],
                [i, i],
                color=colour,
                lw=7,
                alpha=0.30,
                solid_capstyle="butt",
                zorder=2,
            )
            ax.plot([med, med], [i - 0.27, i + 0.27], color="black", lw=2.2, zorder=4)
            ax.text(
                1.02,
                i,
                f"n={len(vals)}   median CHF {med:,.0f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9,
                clip_on=False,
            )

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xscale("log")
        ax.set_xlabel(f"reported cost [CHF, log scale, capped at {CAP_CHF/1e6:.0f}M]")
        ax.set_xlim(right=g["cost_chf"].max() * 1.35)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{asset.title()} costs by damage subtype",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0,
            -0.19,
            f"Specific subtype available for {n_specific}/{n_total} rows; "
            f"{unclassified} unspecified and {generic} only generically described are not plotted.",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
        )

        fig.tight_layout(rect=(0, 0.05, 0.80, 1))
        safe_asset = asset.replace(" ", "_")
        outpath = os.path.join(outdir, f"cost_subtypes_{safe_asset}.png")
        fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"-> {outpath}")


# ------------------------------------------------------------------ plot 2


def plot_convergence(per_asset, outpath, n_draws=200, seed=0):
    """Assess the stability of asset-specific mean costs by repeatedly
    splitting each asset's records into two random halves and comparing
    the means.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for a, g in per_asset.groupby("asset"):
        s = g["cost_chf"].to_numpy()
        if len(s) < 15:
            continue
        diffs = []
        for _ in range(n_draws):
            idx = rng.permutation(len(s))
            h = len(s) // 2
            mean1 = np.mean(s[idx[:h]])
            mean2 = np.mean(s[idx[h:]])
            diffs.append(abs(mean1 - mean2) / max(abs(mean1), abs(mean2)))
        rows.append({"asset": a, "n": len(s), "split_half_diff": np.median(diffs)})

    df = pd.DataFrame(rows).sort_values("n")
    if df.empty:
        return df

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = [COLORS.get(a, "#546e7a") for a in df["asset"]]
    ax.scatter(
        df["n"],
        100 * df["split_half_diff"],
        s=140,
        c=colours,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    for _, r in df.iterrows():
        ax.annotate(
            r["asset"],
            (r["n"], 100 * r["split_half_diff"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.axhline(20, color="#2e7d32", ls="--", lw=1.5, label="20% reference threshold")
    ax.set_xlabel("number of reported cost records")
    ax.set_ylabel("Median difference between random-half means [%]")
    ax.set_xlim(0, df["n"].max() * 1.25)
    ax.set_ylim(0, max(80, 100 * df["split_half_diff"].max() * 1.2))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(
        "How stable are the estimated mean costs?",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")
    return df


# ------------------------------------------------------------------ plot 3


def plot_composition(df, outpath):
    """Two bars: how many figures per category, and how much money."""
    g = df.groupby("asset")["cost_chf"].agg(["count", "sum"])
    g = g.sort_values("count", ascending=True)

    def kind(a):
        if a in PER_ASSET:
            return "per-asset"
        if a in POOLED_ONLY:
            return "pooled only"
        return "excluded"

    palette = {"per-asset": "#d84315", "pooled only": "#f9a825", "excluded": "#90a4ae"}
    colours = [palette[kind(a)] for a in g.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(g) + 3))

    axes[0].barh(g.index, g["count"], color=colours, alpha=0.9)
    for i, v in enumerate(g["count"]):
        axes[0].text(v, i, f"  {v}", va="center", fontsize=9)
    axes[0].set_xlabel("number of reported figures")
    axes[0].set_title("How many figures per category", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, g["count"].max() * 1.2)

    axes[1].barh(g.index, g["sum"] / 1e6, color=colours, alpha=0.9)
    for i, v in enumerate(g["sum"] / 1e6):
        axes[1].text(v, i, f"  {v:,.1f}M", va="center", fontsize=9)
    axes[1].set_xlabel("total reported [CHF millions]")
    axes[1].set_title("How much money per category", fontsize=11, fontweight="bold")
    axes[1].set_xlim(0, (g["sum"] / 1e6).max() * 1.3)
    axes[1].set_yticklabels([])

    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[Patch(facecolor=c, label=k) for k, c in palette.items()],
        loc="lower right",
        fontsize=9,
    )

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "What the reviewed cost figures consist of",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {outpath}")


# --------------------------------------------------------------------- main


def main():
    df = load()
    print(f"  {len(df):,} reviewed figures with a cost")

    above = df["cost_chf"] > CAP_CHF
    print(f"  {above.sum()} above CHF {CAP_CHF:,.0f} - dropped as aggregates")
    df = df[~above]

    per_asset = df[df["asset"].isin(PER_ASSET)]
    pooled = df[~df["asset"].isin(EXCLUDED)]

    print(f"  {len(per_asset):,} name a single asset")
    print(f"  {len(pooled):,} count toward the pooled median")
    for a, why in EXCLUDED.items():
        n = int((df["asset"] == a).sum())
        if n:
            print(f"    excluded {a:20s} n={n:3d}  ({why})")

    # --- the table ---
    t = (
        per_asset.groupby("asset")["cost_chf"]
        .agg(
            n="count",
            p25=lambda s: s.quantile(0.25),
            median="median",
            mean="mean",
            p75=lambda s: s.quantile(0.75),
            max="max",
        )
        .round(0)
    )

    print(f"\n  PER-ASSET COST [CHF]")
    print(t.to_string())

    # A second-level classification based only on wording present in Detail.
    # Unclear / missing descriptions stay explicitly 'unspecified'.
    per_asset = add_subtypes(per_asset)
    subtype_t = (
        per_asset.groupby(["asset", "subtype"])["cost_chf"]
        .agg(n="count", median="median")
        .round(0)
        .sort_index()
    )
    print(f"\n  DETAIL-BASED SUBTYPES [CHF]")
    print(subtype_t.to_string())
    subtype_t.reset_index().to_csv(f"{OUTDIR}/cost_summary_by_subtype.csv", index=False)

    p = pooled["cost_chf"]
    print(
        f"\n  {'pooled':>18} {len(p):5d} {p.quantile(.25):9,.0f} "
        f"{p.median():9,.0f} {p.quantile(.75):9,.0f}"
    )

    t.reset_index().to_csv(f"{OUTDIR}/cost_summary.csv", index=False)

    # --- figures ---
    plot_by_asset(per_asset, f"{OUTDIR}/cost_by_asset.png")
    plot_subtypes_by_asset(per_asset, OUTDIR)

    plot_composition(df, f"{OUTDIR}/cost_composition.png")
    print(f"\n-> {OUTDIR}/cost_summary.csv")
    print(f"-> {OUTDIR}/cost_summary_by_subtype.csv")


if __name__ == "__main__":
    main()
