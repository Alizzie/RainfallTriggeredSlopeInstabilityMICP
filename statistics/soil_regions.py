"""
storme_soil_regions.py — how does USCS soil type distribute across drought regions?

Decides whether soil type can be assigned at REGION level (then applied to the map)
or is too mixed (then keep one soil type, with this analysis as the justification).

It does NOT interpolate soil type (invalid). It only COUNTS which USCS classes occur
in each region and how dominant the top one is (the "purity" metric).

Outputs:
  region_uscs_crosstab.csv     region x USCS counts
  region_soil_summary.csv      region, n, dominant USCS, purity, all types
  storme_points_uscs.csv       x,y,uscs,region  -> load into QGIS as a point layer
  region_uscs_composition.png  stacked bar of the soil mix per region
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

STORME_CSV = "data/wsl_inventory/hangmuren_storme.csv"  # <-- your StorMe file (adjust sep/skiprows)
CALIB = "output/calibration_results.csv"
OUTDIR = "output/soil"
MIN_POINTS = 5  # regions below this = too sparse to trust

# --- columns: ADJUST to your exact StorMe headers ---
# NOTE the swap: in StorMe, "X-Koordinate" is the EASTING (~2.6M) and is ALREADY LV95,
# so do NOT apply the to_lv95(+2e6) shift you use for the big WSL inventory.
COL_EASTING = "X-Koordinate"
COL_NORTHING = "Y-Koordinate"
COL_USCS = "USCS"  # the USCS col of the layer ABOVE the failure zone (section 3.3.1)
COL_ID = "Ereignis-Nr"

VALID_USCS = {
    "GW",
    "GP",
    "GM",
    "GC",
    "SW",
    "SP",
    "SM",
    "SC",
    "ML",
    "CL",
    "OL",
    "MH",
    "CH",
    "OH",
    "PT",
}

os.makedirs(OUTDIR, exist_ok=True)


def clean_uscs(v):
    """'GM (siltiger Kies)' -> 'GM'; keep only valid USCS codes."""
    if pd.isna(v):
        return None
    token = str(v).strip().upper().split()[0].split("(")[0].strip()
    return token if token in VALID_USCS else None


def nearest_region(x, y, calib):
    d = (calib["easting"] - x) ** 2 + (calib["northing"] - y) ** 2
    return int(calib.loc[d.idxmin(), "region_id"])


def main():
    df = pd.read_csv(
        STORME_CSV, skiprows=2
    )  # adjust sep=";" / skiprows / encoding as needed
    calib = pd.read_csv(CALIB)

    df["easting"] = pd.to_numeric(df[COL_EASTING], errors="coerce")
    df["northing"] = pd.to_numeric(df[COL_NORTHING], errors="coerce")
    df["uscs"] = df[COL_USCS].map(clean_uscs)
    df = df.dropna(subset=["easting", "northing", "uscs"])

    if df["easting"].median() < 1_000_000:
        print(
            "WARNING: easting looks 6-digit (LV03) — check COL_EASTING / the X-Y swap!"
        )

    df["region"] = [
        nearest_region(x, y, calib) for x, y in zip(df["easting"], df["northing"])
    ]
    print(f"{len(df)} StorMe points with coordinates + valid USCS")

    ct = pd.crosstab(df["region"], df["uscs"])
    ct.to_csv(f"{OUTDIR}/region_uscs_crosstab.csv")

    rows = []
    for reg, sub in df.groupby("region"):
        counts = sub["uscs"].value_counts()
        n = len(sub)
        rows.append(
            {
                "region": reg,
                "n_points": n,
                "dominant_uscs": counts.index[0],
                "purity": round(counts.iloc[0] / n, 2),
                "n_types": sub["uscs"].nunique(),
                "types": ", ".join(f"{k}:{v}" for k, v in counts.items()),
                "trust": "ok" if n >= MIN_POINTS else "sparse",
            }
        )
    summary = pd.DataFrame(rows).sort_values("n_points", ascending=False)
    summary.to_csv(f"{OUTDIR}/region_soil_summary.csv", index=False)

    df[["easting", "northing", "uscs", "region"]].rename(
        columns={"easting": "x", "northing": "y"}
    ).to_csv(f"{OUTDIR}/storme_points_uscs.csv", index=False)

    trusted = summary[summary["trust"] == "ok"]
    print(f"\nregions with >= {MIN_POINTS} points: {len(trusted)} of {len(summary)}")
    if len(trusted):
        med = trusted["purity"].median()
        print(f"median purity (dominant share) in those: {med:.2f}")
        print("  purity > ~0.70  -> region-level soil assignment is defensible")
        print("  purity low/mixed -> keep one soil type, cite this as the reason")
    print("\nsummary (top rows):")
    print(summary.head(15).to_string(index=False))

    if len(trusted):
        ct_t = ct.loc[ct.index.isin(trusted["region"])]
        shares = ct_t.div(ct_t.sum(axis=1), axis=0)
        ax = shares.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
        ax.set_xlabel("drought region")
        ax.set_ylabel("USCS share")
        ax.set_title(f"soil-type composition per region (>= {MIN_POINTS} points)")
        ax.legend(title="USCS", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/region_uscs_composition.png", dpi=150)
        plt.close()

    print(f"\n-> {OUTDIR}/  (crosstab, summary, points, composition.png)")


if __name__ == "__main__":
    main()
