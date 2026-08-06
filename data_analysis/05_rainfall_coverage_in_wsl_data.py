"""
Check how many inventory events have rainfall data.
Reports which RhiresD years exit, coverage per decade and how many events would be usable including the 120 day spinup before the event.
"""

import sys
import os
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt

from core import data_loader as dl

SPINUP_DAYS = 120
OUTDIR = "output/data_analysis/05_rainfall_coverage_in_wsl_data"
STAT_TXT = f"{OUTDIR}/rainfall_coverage_in_wsl_data.txt"
WSL_INVENTORY_USABLE_CSV = "data/wsl_inventory/wsl_usable_events.csv"
os.makedirs(OUTDIR, exist_ok=True)


def available_years():
    pattern = dl.PATH_RAIN.replace("{}", "*")
    years = set()
    for fp in glob.glob(pattern):
        m = re.search(r"lv95_(\d{4})", os.path.basename(fp))
        if m:
            years.add(int(m.group(1)))
    return years


def main():
    txt = open(STAT_TXT, "w", encoding="utf-8")
    sys.stdout = txt
    sys.stderr = txt

    years = available_years()
    if not years:
        print("No rainfall data found.")
        return
    print(f"RhiresD files: {len(years)} years, {min(years)}–{max(years)}")

    missing = sorted(set(range(min(years), max(years) + 1)) - years)
    if missing:
        print(f"gaps inside that span: {missing}")

    inv = dl.load_wsl_inventory()
    inv["year"] = inv["date"].dt.year
    inv["spin_year"] = (inv["date"] - pd.Timedelta(days=SPINUP_DAYS)).dt.year
    inv["has_event_year"] = inv["year"].isin(years)
    inv["has_spinup"] = inv["spin_year"].isin(years)
    inv["usable"] = inv["has_event_year"] & inv["has_spinup"]

    n = len(inv)
    print(f"\ninventory events: {n}")
    print(
        f"  event year covered : {inv['has_event_year'].sum()} ({inv['has_event_year'].mean():.1%})"
    )
    print(f"  + spin-up covered  : {inv['usable'].sum()} ({inv['usable'].mean():.1%})")
    print(f"  UNUSABLE           : {(~inv['usable']).sum()}")
    print(f"  earliest event {inv['year'].min()} | earliest rainfall {min(years)}")

    dec = (
        inv.assign(decade=(inv["year"] // 10) * 10)
        .groupby("decade")
        .agg(events=("usable", "size"), usable=("usable", "sum"))
    )
    dec["share"] = (dec["usable"] / dec["events"]).round(3)
    print("\nby decade:")
    print(dec.to_string())
    dec.to_csv(f"{OUTDIR}/rainfall_coverage_by_decade.csv")

    # create new csv with only usable events
    usable_events = inv[inv["usable"]]
    usable_events.to_csv(WSL_INVENTORY_USABLE_CSV, index=False)
    print(f"Length of usable events: {len(usable_events)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(dec.index, dec["events"], width=8, color="lightgray", label="all events")
    ax.bar(
        dec.index, dec["usable"], width=8, color="steelblue", label="with rainfall data"
    )
    ax.set_xlabel("decade")
    ax.set_ylabel("events")
    ax.set_title(
        f"Rainfall coverage of the inventory ({inv['usable'].sum()}/{n} usable)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/rainfall_coverage.png", dpi=150)
    plt.close(fig)

    print(f"\n-> {OUTDIR}/rainfall_coverage.png")

    txt.close()


if __name__ == "__main__":
    main()
