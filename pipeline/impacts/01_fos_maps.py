"""
01_fos_maps.py - National FoS maps: bare soil, roots, MICP, and roots + MICP.
Design saturation is the worst case (m_pp=1.0, fixed).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from core import physics
from core import constants as const
from core import data_loader as dl
from pipeline.utils import constants as auct

NONSOIL_TIF = "data/swissTLM3D/tlm_nonsoil_25m_ch.tif"

ROOT_C = {
    12: 15.0,  # closed forest
    14: 10.0,  # dense small stands (copse)
    13: 8.0,  # open forest
    6: 6.0,  # bush forest
}

# literature sensitivity envelope around ROOT_C ("mid"), roughly +-40%.
ROOT_C_SWEEP = {
    "low": {12: 8.0, 14: 6.0, 13: 5.0, 6: 5.0},
    "mid": ROOT_C,
    "high": {12: 22.0, 14: 15.0, 13: 12.0, 6: 8.0},
}

MICP_C = 2.0

M_PP_DESIGN = float(0.79)  # worst case
BETA_MIN, BETA_MAX = auct.BETA_MIN, auct.BETA_MAX
BARE_C = const.C  # calibrated in soil_cohesion.py (~0.0 kPa)
MICP_FEASIBILITY_TIF = "data/injection/binary_injection.tif"

OUTDIR = f"output/impacts/01_maps/m_pp_{M_PP_DESIGN}"
os.makedirs(OUTDIR, exist_ok=True)


def fos_field(beta_deg, m_pp, c_field):
    """FoS for a given slope angle, pore pressure, and cohesion field."""
    beta_rad = np.radians(beta_deg)
    return physics.compute_fos(
        c=c_field,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        beta_rad=beta_rad,
        h_v=const.H_PERP / np.cos(beta_rad),
        m_array=m_pp,
        phi_rad=const.phi,
    )


def required_cohesion(beta_deg, m_pp):
    """Cohesion [kPa] needed to exactly reach FoS = 1 at this pore-pressure ratio."""
    beta_rad = np.radians(beta_deg)
    h_v = const.H_PERP / np.cos(beta_rad)
    tau = const.GAMMA * h_v * np.sin(beta_rad) * np.cos(beta_rad)
    fos0 = physics.compute_fos(
        c=0.0,
        gamma=const.GAMMA,
        gamma_w=const.GAMMA_W,
        h_v=h_v,
        beta_rad=beta_rad,
        phi_rad=const.phi,
        m_array=m_pp,
    )
    return np.maximum((1.0 - fos0) * tau, 0.0)


def read_aligned(path, ref_shape, name):
    """Read a raster and check it matches a reference shape."""
    with rasterio.open(path) as src:
        a = src.read(1)
    if a.shape != ref_shape:
        raise ValueError(
            f"{name} shape {a.shape} does not match reference shape {ref_shape}"
        )
    return a


def cohesion_from(forest, mapping):
    """Root cohesion field from a forest classification raster and a class->kPa mapping."""
    c = np.zeros_like(forest, dtype=np.float32)
    for cls, val in mapping.items():
        c[forest == cls] = val
    return c


def write(beta, valid, profile, name, field, dtype=np.float32, nodata=np.nan):
    """Write one field to OUTDIR/name.tif, filling nodata outside `valid`."""
    out = np.full(beta.shape, nodata, dtype=dtype)
    out[valid] = field
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata)
    with rasterio.open(f"{OUTDIR}/{name}_{M_PP_DESIGN}.tif", "w", **p) as dst:
        dst.write(out, 1)


def write_unstable_area_summary(
    unstable_bare, unstable_root, unstable_micp, unstable_root_micp, n_final
):
    """Write a summary of unstable area statistics to a text file."""
    txt_path = os.path.join(OUTDIR, f"unstable_area_summary_{M_PP_DESIGN}.txt")

    unstable_bare_pct = unstable_bare.mean() * 100
    unstable_root_pct = unstable_root.mean() * 100
    unstable_micp_pct = unstable_micp.mean() * 100
    unstable_root_micp_pct = unstable_root_micp.mean() * 100

    n_bare_unstable = max(unstable_bare.sum(), 1)

    stabilized_root_pct = (unstable_bare & ~unstable_root).sum() / n_bare_unstable * 100

    stabilized_micp_pct = (unstable_bare & ~unstable_micp).sum() / n_bare_unstable * 100

    stabilized_root_micp_pct = (
        (unstable_bare & ~unstable_root_micp).sum() / n_bare_unstable * 100
    )

    with open(txt_path, "w") as file:
        file.write("Factor of Safety - Unstable Area Summary\n")
        file.write("========================================\n\n")

        file.write(f"m_pp = {M_PP_DESIGN}\n")
        file.write(f"Bare-soil cohesion = {BARE_C:.2f} kPa\n")
        file.write(f"MICP cohesion = {MICP_C:.2f} kPa\n")
        file.write(f"Valid soil pixels = {n_final:,}\n\n")

        file.write("Unstable area (FoS < 1)\n")
        file.write("-----------------------\n")
        file.write(f"Bare soil       : {unstable_bare_pct:.2f} %\n")
        file.write(f"Roots           : {unstable_root_pct:.2f} %\n")
        file.write(f"MICP            : {unstable_micp_pct:.2f} %\n")
        file.write(f"Roots + MICP    : {unstable_root_micp_pct:.2f} %\n\n")

        file.write("Share of bare-unstable area stabilized\n")
        file.write("--------------------------------------\n")
        file.write(f"Roots           : {stabilized_root_pct:.2f} %\n")
        file.write(f"MICP            : {stabilized_micp_pct:.2f} %\n")
        file.write(f"Roots + MICP    : {stabilized_root_micp_pct:.2f} %\n")

    print(f"summary saved to {txt_path}")


def write_micp_landslide_class(beta, valid, profile, micp_possible, unstable_bare):
    """
    Combine MICP feasibility and bare-soil instability.

    Classes:
        0 = MICP no  + unstable no
        1 = MICP no  + unstable yes
        2 = MICP yes + unstable no
        3 = MICP yes + unstable yes

    Encoding:
        class = 2 * MICP_possible + unstable
    """

    micp = micp_possible.astype(np.uint8)
    landslide = unstable_bare.astype(np.uint8)

    categories = 2 * micp + landslide
    # Check category distribution
    counts = np.bincount(categories, minlength=4)

    print("\nMICP x landslide category check:")
    print(f"0 | MICP no  + stable   : {counts[0]:,}")
    print(f"1 | MICP no  + unstable : {counts[1]:,}")
    print(f"2 | MICP yes + stable   : {counts[2]:,}")
    print(f"3 | MICP yes + unstable : {counts[3]:,}")
    print(f"Unique category values  : {np.unique(categories)}")

    # Category Statistics
    counts = np.bincount(categories, minlength=4)
    n_total = categories.size
    percentages = counts / n_total * 100
    n0, n1, n2, n3 = counts
    n_unstable = n1 + n3

    if n_unstable > 0:
        unstable_micp_yes_pct = n3 / n_unstable * 100
        unstable_micp_no_pct = n1 / n_unstable * 100
    else:
        unstable_micp_yes_pct = 0.0
        unstable_micp_no_pct = 0.0

    # Write to TIFF
    nodata = 255
    out = np.full(beta.shape, nodata, dtype=np.uint8)
    out[valid] = categories

    p = profile.copy()
    p.update(dtype=np.uint8, nodata=nodata, count=1)

    out_path = os.path.join(OUTDIR, f"micp_landslide_categories_{M_PP_DESIGN}.tif")

    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(out, 1)

    # Wirte Txt summary
    txt_path = os.path.join(
        OUTDIR, f"micp_landslide_categories_summary_{M_PP_DESIGN}.txt"
    )

    labels = {
        0: "MICP not feasible | stable",
        1: "MICP not feasible | unstable",
        2: "MICP feasible     | stable",
        3: "MICP feasible     | unstable",
    }

    with open(txt_path, "w") as file:

        file.write("MICP Feasibility and Landslide Instability Summary\n")
        file.write("===================================================\n\n")

        file.write(f"m_pp = {M_PP_DESIGN}\n")
        file.write(f"Valid soil pixels = {n_total:,}\n\n")

        file.write("Category definitions\n")
        file.write("--------------------\n")

        file.write("0 = MICP not feasible + stable\n")
        file.write("1 = MICP not feasible + unstable\n")
        file.write("2 = MICP feasible + stable\n")
        file.write("3 = MICP feasible + unstable\n\n")

        file.write("Distribution of MICP feasibility and " "landslide instability\n")
        file.write("--------------------------------------------------\n")

        for cls in range(4):

            file.write(
                f"{cls} - {labels[cls]:32s}: "
                f"{counts[cls]:,} pixels "
                f"({percentages[cls]:.2f} %)\n"
            )

        file.write("\n")
        file.write("Landslide-unstable area\n")
        file.write("-----------------------\n")

        file.write(f"Total unstable pixels: {n_unstable:,}\n")

        file.write(
            f"MICP feasible     : {n3:,} pixels " f"({unstable_micp_yes_pct:.2f} %)\n"
        )

        file.write(
            f"MICP not feasible : {n1:,} pixels " f"({unstable_micp_no_pct:.2f} %)\n"
        )

        file.write("\n")

        file.write(
            "MICP-coverable share of landslide-unstable area: "
            f"{unstable_micp_yes_pct:.2f} %\n"
        )

    # Pie Chart
    if n_unstable > 0:

        fig, ax = plt.subplots(figsize=(7, 7))

        values = [
            n3,
            n1,
        ]

        pie_labels = [
            "MICP feasible",
            "MICP not feasible",
        ]

        ax.pie(
            values,
            labels=pie_labels,
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title("MICP coverable share of landslide-unstable area")

        ax.axis("equal")

        fig.tight_layout()

        pie1_path = os.path.join(
            OUTDIR,
            f"pie_micp_coverable_landslide_area_{M_PP_DESIGN}.png",
        )

        fig.savefig(
            pie1_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))

    pie_labels = [
        "MICP not feasible\nStable",
        "MICP not feasible\nUnstable",
        "MICP feasible\nStable",
        "MICP feasible\nUnstable",
    ]

    ax.pie(
        counts,
        labels=pie_labels,
        autopct="%1.1f%%",
        startangle=90,
    )

    ax.set_title("Distribution of MICP Feasibility and " "Landslide Instability")

    ax.axis("equal")

    fig.tight_layout()

    pie2_path = os.path.join(
        OUTDIR,
        f"pie_micp_landslide_categories_{M_PP_DESIGN}.png",
    )

    fig.savefig(
        pie2_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"MICP-landslide categories saved to {out_path}")


def write_rescue_categories(
    beta,
    valid,
    profile,
    unstable_bare,
    unstable_root,
    unstable_micp,
    unstable_root_micp,
):
    """
    Classify initially unstable bare-soil pixels according to
    which intervention stabilizes them.

    Classes:
        1 = rescued by MICP only
        2 = rescued by roots only
        3 = rescued by both MICP and roots independently
        4 = rescued only by roots + MICP combined
        5 = still unstable after roots + MICP

    Stable bare-soil pixels are NoData (255).
    """

    baseline = unstable_bare

    stable_micp = ~unstable_micp
    stable_root = ~unstable_root
    stable_combined = ~unstable_root_micp

    rescued_micp_only = baseline & stable_micp & ~stable_root
    rescued_root_only = baseline & stable_root & ~stable_micp
    rescued_both = baseline & stable_micp & stable_root
    rescued_combined_only = baseline & unstable_micp & unstable_root & stable_combined
    still_unstable = baseline & unstable_root_micp

    # Create categorical array

    nodata = 255

    categories = np.full(
        baseline.shape,
        nodata,
        dtype=np.uint8,
    )

    categories[rescued_micp_only] = 1
    categories[rescued_root_only] = 2
    categories[rescued_both] = 3
    categories[rescued_combined_only] = 4
    categories[still_unstable] = 5

    classified = (
        rescued_micp_only
        | rescued_root_only
        | rescued_both
        | rescued_combined_only
        | still_unstable
    )

    n_baseline = int(baseline.sum())
    n_classified = int(classified.sum())

    if n_classified != n_baseline:
        raise ValueError(
            "Rescue classification error: "
            f"{n_classified:,} classified vs "
            f"{n_baseline:,} initially unstable."
        )

    # Statistics
    counts = {
        1: int(rescued_micp_only.sum()),
        2: int(rescued_root_only.sum()),
        3: int(rescued_both.sum()),
        4: int(rescued_combined_only.sum()),
        5: int(still_unstable.sum()),
    }

    percentages = {
        cls: count / max(n_baseline, 1) * 100 for cls, count in counts.items()
    }

    labels = {
        1: "Rescued by MICP only",
        2: "Rescued by roots only",
        3: "Rescued by both independently",
        4: "Rescued by roots + MICP only",
        5: "Still unstable",
    }

    # Write GeoTIFF
    out = np.full(
        beta.shape,
        nodata,
        dtype=np.uint8,
    )

    out[valid] = categories

    p = profile.copy()

    p.update(
        dtype="uint8",
        nodata=nodata,
        count=1,
    )

    tif_path = os.path.join(
        OUTDIR,
        f"rescue_categories_{M_PP_DESIGN}.tif",
    )

    with rasterio.open(tif_path, "w", **p) as dst:
        dst.write(out, 1)

    # Overall rescue statistics
    total_rescued = counts[1] + counts[2] + counts[3] + counts[4]

    total_rescued_pct = total_rescued / max(n_baseline, 1) * 100

    # Write TXT
    txt_path = os.path.join(
        OUTDIR,
        f"rescue_categories_summary_{M_PP_DESIGN}.txt",
    )

    with open(txt_path, "w") as file:

        file.write("Landslide Stabilization Outcome Summary\n")
        file.write("=======================================\n\n")

        file.write(f"m_pp = {M_PP_DESIGN}\n")
        file.write(f"MICP cohesion = {MICP_C:.2f} kPa\n")
        file.write(f"Initially unstable pixels = " f"{n_baseline:,}\n\n")

        file.write("Share of initially unstable area\n")
        file.write("--------------------------------\n")

        for cls in range(1, 6):

            file.write(
                f"{cls} - {labels[cls]:32s}: "
                f"{counts[cls]:,} pixels "
                f"({percentages[cls]:.2f} %)\n"
            )

        file.write("\n")

        file.write(
            f"Total rescued                 : "
            f"{total_rescued:,} pixels "
            f"({total_rescued_pct:.2f} %)\n"
        )

        file.write(
            f"Still unstable                : "
            f"{counts[5]:,} pixels "
            f"({percentages[5]:.2f} %)\n"
        )

    # Console output
    print("\nLandslide stabilization outcome:")
    print("--------------------------------")

    for cls in range(1, 6):

        print(
            f"{cls} | {labels[cls]:32s}: "
            f"{counts[cls]:,} "
            f"({percentages[cls]:.2f} %)"
        )

    print("")
    print(f"Total rescued: {total_rescued:,} " f"({total_rescued_pct:.2f} %)")

    print(f"Still unstable: {counts[5]:,} " f"({percentages[5]:.2f} %)")

    print(f"\nRescue category TIFF saved to {tif_path}")
    print(f"Rescue summary saved to {txt_path}")


def plot_required_cohesion_histograms(
    c_req,
    c_req_root,
    c_req_micp,
    c_req_root_micp,
):
    """
    Save one required-cohesion histogram per treatment scenario.

    Classes:
        <= 1 kPa
        1-2 kPa
        2-3 kPa
        3-5 kPa
        5-7 kPa
        > 7 kPa
    """

    bins = [-np.inf, 1, 2, 3, 5, 7, np.inf]

    bin_labels = [
        "≤1",
        "1-2",
        "2-3",
        "3-5",
        "5-7",
        ">7",
    ]

    scenarios = {
        "bare_soil": ("Bare", c_req),
        "roots": ("Roots", c_req_root),
        "micp": ("MICP", c_req_micp),
        "roots_micp": ("Roots + MICP", c_req_root_micp),
    }

    for filename, (title, values) in scenarios.items():

        values = np.asarray(values)
        values = values[np.isfinite(values)]

        # Number of pixels in each cohesion class
        counts, _ = np.histogram(values, bins=bins)

        # Percentage of pixels
        percentages = counts / max(counts.sum(), 1) * 100

        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(bin_labels))

        bars = ax.bar(
            x,
            percentages,
            width=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)

        ax.set_xlabel("Required additional cohesion [kPa]")
        ax.set_ylabel("Share of soil pixels [%]")

        ax.set_title(f"Required Cohesion ({title})\n" f"$m_{{pp}}$ = {M_PP_DESIGN:g}")

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        # Add percentage above each bar
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()

        out_path = os.path.join(
            OUTDIR,
            f"required_cohesion_histogram_{filename}_{M_PP_DESIGN}.png",
        )

        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Required-cohesion histogram saved to {out_path}")


def plot_baseline_fos_unstable(fos_bare, unstable_bare):
    """
    Plot the baseline Factor of Safety distribution for pixels that
    are unstable under bare-soil conditions (FoS < 1).
    """

    fos_unstable = np.asarray(fos_bare[unstable_bare])
    fos_unstable = fos_unstable[np.isfinite(fos_unstable)]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        fos_unstable,
        bins=40,
        edgecolor="black",
        linewidth=0.5,
    )

    # Stability threshold
    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Stability threshold (FoS = 1)",
    )

    ax.set_xlabel("Baseline factor of safety, FoS [-]")
    ax.set_ylabel("Number of unstable pixels")

    ax.set_title(
        f"Baseline FoS Distribution of Unstable Pixels\n"
        f"$m_{{pp}}$ = {M_PP_DESIGN:g}"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    out_path = os.path.join(
        OUTDIR,
        f"baseline_fos_unstable_distribution_{M_PP_DESIGN}.png",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Baseline FoS distribution saved to {out_path}")


def main():
    with rasterio.open(auct.SLOPE_TIF) as src:
        beta = src.read(1).astype(np.float32)
        profile = src.profile
        slope_nd = src.nodata

    forest, _, _ = dl.load_forest_raster()
    if forest.shape != beta.shape:
        raise ValueError(f"forest shape {forest.shape} != slope shape {beta.shape}")
    nonsoil = read_aligned(NONSOIL_TIF, beta.shape, "non-soil")
    micp_possible = read_aligned(MICP_FEASIBILITY_TIF, beta.shape, "MICP feasibility")

    valid = np.isfinite(beta)
    if slope_nd is not None:
        valid &= beta != slope_nd

    valid &= (beta >= BETA_MIN) & (beta <= BETA_MAX)
    n_slope = int(valid.sum())
    valid &= nonsoil != 1
    n_final = int(valid.sum())
    print(f"pixels in {BETA_MIN:.0f}-{BETA_MAX:.0f} deg: {n_slope:,}")
    print(
        f"  removed as rock/glacier/water/snow: {n_slope - n_final:,} "
        f"({1 - n_final / max(n_slope, 1):.1%})"
    )
    print(f"  soil-mantled pixels used: {n_final:,}")

    b = beta[valid]
    f = forest[valid]
    micp_feasible = micp_possible[valid] == 1

    values, value_counts = np.unique(
        micp_possible,
        return_counts=True,
    )

    print("\nRaw binary_injection.tif values:")
    for value, count in zip(values, value_counts):
        print(f"  {value}: {count:,}")

    c_root = cohesion_from(f, ROOT_C)
    c_micp = np.where(micp_feasible, MICP_C, 0.0).astype(np.float32)

    fos_bare = fos_field(b, M_PP_DESIGN, BARE_C)
    fos_root = fos_field(b, M_PP_DESIGN, BARE_C + c_root)
    fos_micp = fos_field(b, M_PP_DESIGN, BARE_C + c_micp)
    fos_root_micp = fos_field(b, M_PP_DESIGN, BARE_C + c_root + c_micp)

    c_req = required_cohesion(b, M_PP_DESIGN)
    r_req_root = np.maximum(c_req - c_root, 0.0)
    c_req_micp = np.maximum(c_req - c_micp, 0.0)
    c_req_root_micp = np.maximum(c_req - c_root - c_micp, 0.0)

    plot_required_cohesion_histograms(
        c_req,
        r_req_root,
        c_req_micp,
        c_req_root_micp,
    )

    write(beta, valid, profile, "fos_bare", fos_bare)
    write(beta, valid, profile, "fos_root", fos_root)
    write(beta, valid, profile, "fos_micp", fos_micp)
    write(beta, valid, profile, "fos_root_micp", fos_root_micp)

    write(beta, valid, profile, "required_cohesion", c_req)
    write(beta, valid, profile, "required_cohesion_above_roots", r_req_root)
    write(beta, valid, profile, "required_cohesion_above_micp", c_req_micp)
    write(
        beta, valid, profile, "required_cohesion_above_roots_and_micp", c_req_root_micp
    )

    unstable_bare = fos_bare < 1
    unstable_root = fos_root < 1
    unstable_micp = fos_micp < 1
    unstable_root_micp = fos_root_micp < 1

    plot_baseline_fos_unstable(
        fos_bare,
        unstable_bare,
    )

    write_micp_landslide_class(beta, valid, profile, micp_feasible, unstable_bare)

    print(f"forested share of soil pixels: {(c_root > 0).mean():.1%}")
    print(f"MICP additional cohesion: {MICP_C:.2f} kPa")

    print(f"unstable (FoS < 1) at m_pp={M_PP_DESIGN:g}, c_bare={BARE_C:.2f} kPa:")
    print(f"bare soil               : {unstable_bare.mean():.1%}")
    print(f"+ roots                 : {unstable_root.mean():.1%}")
    print(f"+ MICP ({MICP_C:g} kPa)       : {unstable_micp.mean():.1%}")
    print(f"+ roots + MICP          : {unstable_root_micp.mean():.1%}")

    print("")
    print("share of bare-unstable pixels stabilized:")
    print(
        f"roots        : {(unstable_bare & ~unstable_root).sum() / max(unstable_bare.sum(), 1):.1%}"
    )
    print(
        f"MICP         : {(unstable_bare & ~unstable_micp).sum() / max(unstable_bare.sum(), 1):.1%}"
    )
    print(
        f"roots + MICP : {(unstable_bare & ~unstable_root_micp).sum() / max(unstable_bare.sum(), 1):.1%}"
    )

    print("")
    print(
        f"required cohesion: median={np.median(c_req):.2f}, p90={np.percentile(c_req, 90):.2f}, max={c_req.max():.2f} kPa"
    )

    print("")
    print("sensitivity to the root-cohesion assumption:")
    print(f"{'variant':>8} {'unstable_root':>15} {'unstable_root+micp':>20}")

    for name, mapping in ROOT_C_SWEEP.items():
        cr = cohesion_from(f, mapping)
        fos_r = fos_field(b, M_PP_DESIGN, BARE_C + cr)
        fos_rm = fos_field(b, M_PP_DESIGN, BARE_C + cr + c_micp)
        print(f"{name:>8} {(fos_r < 1).mean():14.1%} {(fos_rm < 1).mean():19.1%}")

    print("")
    print(f"results saved to {OUTDIR}")
    write_unstable_area_summary(
        unstable_bare, unstable_root, unstable_micp, unstable_root_micp, n_final
    )
    write_rescue_categories(
        beta,
        valid,
        profile,
        unstable_bare,
        unstable_root,
        unstable_micp,
        unstable_root_micp,
    )


if __name__ == "__main__":
    main()
