"""
saturation_animation.py - Stitches the PNG frames produced by
map_saturation_sweep.py into:

    1. saturation_sweep.gif       - animated GIF, ~1.2s per frame (slow enough to read)
    2. saturation_sweep.mp4       - same animation as video (smoother, scrubbable)
    3. saturation_filmstrip.png   - all frames in a static grid, no animation needed

Reads the PNGs already saved by map_saturation_sweep.py — never touches the
raw rasters again, so this step is cheap and disk-light (only reads/writes
images that already exist on disk).

Run map_saturation_sweep.py first.
"""

import os
import glob
import re
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- config

FRAMES_DIR = "output/impacts/03_saturation_sweep/bare/frames"
OUTDIR = "output/impacts/03_saturation_sweep/bare/animation"
os.makedirs(OUTDIR, exist_ok=True)

FIELD = "fos_bare"  # must match --field used in map_saturation_sweep.py

# How long each frame is shown, in milliseconds. Default GIF playback is often
# ~100ms/frame (too fast to read); this is deliberately slower.
FRAME_DURATION_MS = 1200


# ------------------------------------------------------------------------ helpers


def find_frame_pngs(field):
    """Locate all {field}_mpp{X}.png frames, sorted by m_pp value."""
    pattern = f"{FRAMES_DIR}/{field}_mpp*.png"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No files found matching {pattern} - run map_saturation_sweep.py first "
            f"(with --field {field}, if not using the default)."
        )

    def m_pp_of(path):
        m = re.search(r"mpp([\d.]+)\.png$", path)
        return float(m.group(1))

    files = sorted(files, key=m_pp_of)
    return [(m_pp_of(p), p) for p in files]


# ---------------------------------------------------------------------- animated GIF


def make_gif(frame_files):
    """Slow-paced animated GIF directly from the saved PNG frames."""
    images = [Image.open(p).convert("RGB") for _, p in frame_files]

    gif_path = f"{OUTDIR}/saturation_sweep.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,  # ms per frame
        loop=0,  # loop forever
    )
    print(f"-> {gif_path}  ({FRAME_DURATION_MS} ms/frame)")
    return images


def make_mp4(frame_files):
    """Optional MP4 version - smoother, scrubbable, no GIF frame-rate quirks."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            "[skip] imageio not installed - mp4 not created (pip install imageio imageio-ffmpeg)"
        )
        return

    fps = 1000 / FRAME_DURATION_MS
    mp4_path = f"{OUTDIR}/saturation_sweep.mp4"
    try:
        with imageio.get_writer(mp4_path, fps=fps) as writer:
            for _, p in frame_files:
                writer.append_data(imageio.imread(p))
        print(f"-> {mp4_path}")
    except Exception as e:
        print(f"[skip] mp4 not created ({e})")


# ---------------------------------------------------------------------- filmstrip


def make_filmstrip(frame_files, ncols=4):
    """All frames in a static grid - no animation needed, good for a figure/poster."""
    n = len(frame_files)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows), squeeze=False
    )

    for i, (m_pp, path) in enumerate(frame_files):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"m_pp = {m_pp:.1f}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Saturation sweep — {FIELD}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = f"{OUTDIR}/saturation_filmstrip.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


# ---------------------------------------------------------------------------- main


def main():
    frame_files = find_frame_pngs(FIELD)
    print(f"Found {len(frame_files)} frames for field '{FIELD}':")
    for m_pp, path in frame_files:
        print(f"  m_pp={m_pp:.1f}  <-  {path}")
    print()

    make_gif(frame_files)
    make_mp4(frame_files)
    make_filmstrip(frame_files)

    print(f"\n✓ All outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
