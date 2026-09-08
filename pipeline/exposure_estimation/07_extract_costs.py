"""
07_extract_costs - List the events that mention money.
"""

import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from core import data_loader as dl

# ============================================================== CONFIG =====

OUTDIR = "output/costs"
os.makedirs(OUTDIR, exist_ok=True)

# Set to None for every canton.
ALPINE_CANTONS = [
    "GR",
    "VS",
    "BE",
    "UR",
    "TI",
    "GL",
    "OW",
    "NW",
    "SZ",
    "SG",
    "AI",
    "AR",
    "VD",
    "LU",
    "FR",
]

COL_PROCESS = "main process"
COL_DESC = "description of the event"
COL_CANTON = "canton (country subdivision)"
COL_AREA = "name of area"
COL_DAMAGE = (
    "extent of damage, low [0.01-0.4], medium [0.4-2], "
    "large/catastrophic [>2] or death [mio. CHF]"
)

# Anything that looks like a sum of money, with a little text around it.
MONEY = re.compile(
    r".{0,40}(?:"
    r"(?:CHF|Fr\.?|Franken)\s*[\d'’\s.,]{3,15}"
    r"|[\d'’\s.,]{3,15}\s*(?:CHF|Fr\.?|Franken|\.-)"
    r"|[\d.,]+\s*(?:Mio\.?|Millionen)"
    r").{0,40}",
    re.I,
)

# Excel caps a cell at 32,767 characters; a few descriptions run longer.
TEXT_LIMIT = 30_000

# =========================================================== END CONFIG ====


def fix_encoding(text):
    """Repair UTF-8 text that was read with the wrong codec.

    The WSL export shows "versch√ºttet" where it should read "verschuettet",
    which is UTF-8 decoded as MacRoman. Tries MacRoman first, then Latin-1.
    """
    if not isinstance(text, str) or ("√" not in text and "Ã" not in text):
        return text
    for codec in ("mac_roman", "latin-1"):
        try:
            fixed = text.encode(codec, errors="ignore").decode("utf-8", errors="ignore")
            if "√" not in fixed and "Ã" not in fixed:
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def money_snippets(text):
    """Every money-looking phrase in one description, joined for easy reading."""
    hits = [" ".join(m.group(0).split()) for m in MONEY.finditer(text)]
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return "  |  ".join(out)


def main():
    df = dl.load_wsl_inventory()
    n_all = len(df)

    df = df[df[COL_PROCESS].astype(str).str.contains("Rutschung", na=False)]
    print(f"  {len(df):,}/{n_all:,} events are landslides")

    if ALPINE_CANTONS and COL_CANTON in df.columns:
        before = len(df)
        df = df[df[COL_CANTON].astype(str).str.strip().isin(ALPINE_CANTONS)]
        print(f"  {len(df):,}/{before:,} are in alpine cantons")

    df = df.copy()
    df["description"] = df[COL_DESC].map(fix_encoding).fillna("")

    # Many WSL rows repeat the same description under different area names,
    # so the text is compared from the end, where the shared block sits.
    before = len(df)
    df = df.assign(_tail=df["description"].str[-800:])
    df = df.drop_duplicates(subset=["_tail"]).drop(columns="_tail")
    print(f"  {len(df):,}/{before:,} after dropping repeated descriptions")

    df["money_mentions"] = df["description"].map(money_snippets)
    money = df[df["money_mentions"] != ""].copy()
    print(f"\n  {len(money):,} events mention money")

    out = pd.DataFrame(
        {
            "date": money["date"].dt.date,
            "canton": money[COL_CANTON],
            "municipality": money.get("municipality"),
            "area": money.get(COL_AREA),
            "damage_class": money[COL_DAMAGE],
            "money_mentions": money["money_mentions"],
            # columns for you to fill in while reading
            "cost_chf": "",
            "asset": "",
            "is_whole_storm_total": "",
            "note": "",
            "description": money["description"].str[:TEXT_LIMIT],
        }
    ).sort_values("date")

    path = f"{OUTDIR}/money_events.csv"
    out.to_csv(path, index=False)

    print(f"\n  by damage class:")
    for cls, n in money[COL_DAMAGE].value_counts().items():
        print(f"    {str(cls):>22}: {n:,}")

    print(f"\n  by canton:")
    for canton, n in money[COL_CANTON].value_counts().head(10).items():
        print(f"    {str(canton):>22}: {n:,}")

    print(f"\n-> {path}")
    print("\n  Read `money_mentions` first - it lists just the amounts. Open")
    print("  `description` only when the context is unclear. Fill in cost_chf,")
    print("  asset and is_whole_storm_total as you go.")


if __name__ == "__main__":
    main()
