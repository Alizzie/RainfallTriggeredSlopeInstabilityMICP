"""
compare_silvaprotect.py - cross-tabulate our FoS map against the BAFU SilvaProtect
Hangmuren map.

QGIS PREP:
  1. Load the SilvaProtect Hangmuren GDB, inspect the attribute table (hazard index? class?).
  2. Processing > GDAL > Rasterize: burn 1 for hazard polygons, NoData 0, onto EXACTLY
     the grid of your FoS raster -> data/silvaprotect_hangmuren.tif
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOTS_TIF = "output/impacts/01_maps/fos_root.tif"
FOS_TIF = "output/impacts/01_maps/fos_bare.tif"
SP_TIF = "data/hangmuren_silverprotect/hangmuren_processed.tif"
OUTDIR = "output/impacts/02_silverprotect_comparison"
FOS_THRESHOLD = 1.0
os.makedirs(OUTDIR, exist_ok=True)


def main():

    for name, path in [("FoS", FOS_TIF), ("FoS + roots", ROOTS_TIF)]:
        with rasterio.open(path) as src:
            fos = src.read(1).astype("float32")
            profile = src.profile
        with rasterio.open(SP_TIF) as src:
            sp = src.read(1)
        if sp.shape != fos.shape:
            raise SystemExit(
                f"grid mismatch: FoS {fos.shape} vs SilvaProtect {sp.shape} — "
                f"re-rasterize onto the FoS grid."
            )

        valid = np.isfinite(fos)
        ours = (fos < FOS_THRESHOLD) & valid
        theirs = (sp == 1) & valid

        both = int((ours & theirs).sum())
        only_ours = int((ours & ~theirs).sum())
        only_theirs = int((~ours & theirs).sum())
        neither = (
            int((~ours & ~theirs).sum() & 1)
            if False
            else int((valid & ~ours & ~theirs).sum())
        )
        n = int(valid.sum())

        agree = (both + neither) / n
        # Jaccard on the "unstable" class only — more informative than overall agreement,
        # which is inflated by the large stable background.
        union = both + only_ours + only_theirs
        jaccard = both / union if union else np.nan

        with open(
            f"{OUTDIR}/agreement_silvaprotect_{name.replace(' ', '_')}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"valid pixels: {n:,}\n")
            f.write(f"\n{'':>22}{'SP unstable':>14}{'SP stable':>12}\n")
            f.write(f"{'ours unstable':>22}{both:>14,}{only_ours:>12,}\n")
            f.write(f"{'ours stable':>22}{only_theirs:>14,}{neither:>12,}\n")
            f.write(f"\nour unstable share  : {ours.sum() / n:.1%}\n")
            f.write(f"their unstable share: {theirs.sum() / n:.1%}\n")
            f.write(f"overall agreement   : {agree:.1%}\n")
            f.write(f"Jaccard (unstable)  : {jaccard:.3f}\n")
            if theirs.sum():
                f.write(f"of their unstable, we also flag: {both / theirs.sum():.1%}\n")
            if ours.sum():
                f.write(f"of our unstable, they also flag: {both / ours.sum():.1%}\n")

        # agreement raster: 0 neither, 1 only ours, 2 only theirs, 3 both
        agree_map = np.zeros(fos.shape, dtype="uint8")
        agree_map[valid & ours & ~theirs] = 1
        agree_map[valid & ~ours & theirs] = 2
        agree_map[valid & ours & theirs] = 3
        p = profile.copy()
        p.update(dtype="uint8", count=1, nodata=255)
        agree_map[~valid] = 255
        with rasterio.open(
            f"{OUTDIR}/agreement_silvaprotect_{name.replace(' ', '_')}.tif", "w", **p
        ) as dst:
            dst.write(agree_map, 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["neither", "only ours", "only SilvaProtect", "both"]
        vals = [neither, only_ours, only_theirs, both]
        ax.bar(
            labels,
            np.array(vals) / n * 100,
            color=["lightgray", "firebrick", "steelblue", "purple"],
        )
        ax.set_ylabel("% of valid pixels")
        ax.set_title(f"Agreement with SilvaProtect (Jaccard {jaccard:.3f})")
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(
            f"{OUTDIR}/agreement_silvaprotect_{name.replace(' ', '_')}.png", dpi=150
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
