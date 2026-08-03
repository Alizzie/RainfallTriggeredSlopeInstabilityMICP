"""Draw the BAFU drought regions on the Swiss map: one colour per region, region ID inside each area."""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import data_loader as dl
from core import region_map as rm

OUTDIR = "output/data_analysis/00_map_regions"
OUTPUT_FILE = f"{OUTDIR}/drought_regions.png"
os.makedirs(OUTDIR, exist_ok=True)


def main():
    """Render every drought-region polygon colourised, with its ID inside."""
    geometries = dl.load_region_geometries()
    label_points = {rid: dl.region_representative_point(rid) for rid in geometries}

    fig, _ = rm.plot_region_id_map(
        geometries,
        label_points=label_points,
        title="BAFU drought regions",
        cmap="tab20",
        axis_off=True,
    )
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Region map written to {OUTPUT_FILE} ({len(geometries)} regions).")


if __name__ == "__main__":
    main()
