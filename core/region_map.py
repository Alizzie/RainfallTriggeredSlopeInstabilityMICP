"""
region_map.py — Reusable choropleth mapping for the BAFU drought regions.

Fills each drought-region polygon by a per-region value, outlines the borders
as thin lines and writes the region ID inside every region. One static map, one
animation over an ordered sequence of frames (e.g. the 12 months).

Depends only on numpy + matplotlib: the region shapes are read straight from the
raw GeoJSON geometry dicts returned by ``data_loader.load_region_geometries``,
so no GeoPandas / shapely is required for plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import PathPatch
from matplotlib.path import Path

MISSING_COLOR = "lightgray"  # regions without a value
BORDER_KW = {"edgecolor": "#333333", "linewidth": 0.4}  # thin region outlines
LABEL_KW = {"fontsize": 6, "color": "black", "ha": "center", "va": "center"}


# ---------------------------------------------------------------------
# GeoJSON -> matplotlib geometry
# ---------------------------------------------------------------------


def _iter_parts(geometry: dict):
    """Yield (exterior_ring, [hole_rings]) for each part of a (Multi)Polygon."""
    kind = geometry["type"]
    if kind == "Polygon":
        rings = [geometry["coordinates"]]
    elif kind == "MultiPolygon":
        rings = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {kind}")
    for part in rings:
        yield part[0], part[1:]


def _signed_area(ring) -> float:
    """Signed area of a ring (shoelace); positive means counter-clockwise."""
    pts = np.asarray(ring, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(ring, ccw: bool):
    """Return the ring wound counter-clockwise (ccw=True) or clockwise."""
    pts = np.asarray(ring, dtype=float)
    if (_signed_area(pts) < 0) == ccw:  # wrong winding -> reverse
        pts = pts[::-1]
    return pts


def geometry_to_path(geometry: dict) -> Path:
    """
    Build one compound matplotlib Path for a whole (Multi)Polygon.

    Exteriors are wound counter-clockwise and holes clockwise so the non-zero
    fill rule renders holes as holes.
    """
    vertices, codes = [], []
    for exterior, holes in _iter_parts(geometry):
        for ring, ccw in [(exterior, True)] + [(h, False) for h in holes]:
            pts = _orient(ring, ccw)
            if not np.array_equal(pts[0], pts[-1]):  # close the ring
                pts = np.vstack([pts, pts[0]])
            vertices.append(pts)
            ring_codes = np.full(len(pts), Path.LINETO, dtype=np.uint8)
            ring_codes[0] = Path.MOVETO
            ring_codes[-1] = Path.CLOSEPOLY
            codes.append(ring_codes)
    return Path(np.concatenate(vertices), np.concatenate(codes))


def _ring_centroid(pts: np.ndarray) -> tuple[float, float]:
    """Area-weighted centroid of a closed ring (shoelace formula)."""
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * np.sum(cross)
    if np.isclose(a, 0):  # degenerate -> plain mean
        return float(x.mean()), float(y.mean())
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * a)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * a)
    return float(cx), float(cy)


def geometry_label_point(geometry: dict) -> tuple[float, float]:
    """
    Fallback label anchor: area-weighted centroid of the largest part's
    exterior. Pass explicit ``label_points`` for a guaranteed-inside anchor
    (e.g. ``data_loader.region_representative_point``).
    """
    best_pts, best_area = None, -1.0
    for exterior, _ in _iter_parts(geometry):
        pts = np.asarray(exterior, dtype=float)
        area = abs(_signed_area(pts))
        if area > best_area:
            best_pts, best_area = pts, area
    return _ring_centroid(best_pts)


def part_area_weights(geometry: dict) -> list[tuple[tuple[float, float], float]]:
    """
    [(centroid_xy, area_weight), ...] for every part of a (Multi)Polygon,
    weights summing to 1 (holes subtracted from their part's area). A plain
    Polygon returns a single entry with weight 1.0 — safe to call on every
    region, multipart or not.

    Use where a single representative point would under-represent a region
    made of several disconnected areas (e.g. sampling a gridded variable
    like rainfall once per part and area-weighting the result), since
    ``geometry_label_point`` / ``region_representative_point`` only look at
    the largest part.
    """
    parts = []
    for exterior, holes in _iter_parts(geometry):
        ext_pts = np.asarray(exterior, dtype=float)
        area = abs(_signed_area(ext_pts))
        for hole in holes:
            area -= abs(_signed_area(np.asarray(hole, dtype=float)))
        parts.append((_ring_centroid(ext_pts), max(area, 0.0)))

    total = sum(area for _, area in parts)
    if total <= 0:  # degenerate geometry -> equal weights
        n = len(parts)
        return [(pt, 1.0 / n) for pt, _ in parts]
    return [(pt, area / total) for pt, area in parts]


# ---------------------------------------------------------------------
# Static drawing helpers
# ---------------------------------------------------------------------


def _resolve_label_points(geometries, label_points):
    """Use caller-supplied anchors where given, else the centroid fallback."""
    label_points = label_points or {}
    return {
        rid: label_points.get(rid, geometry_label_point(geom))
        for rid, geom in geometries.items()
    }


def draw_region_borders(
    ax, geometries, *, label_points=None, label_ids=True, border_kw=None, label_kw=None
):
    """Outline every region with thin lines and (optionally) write its ID."""
    border_kw = {**BORDER_KW, **(border_kw or {})}
    label_kw = {**LABEL_KW, **(label_kw or {})}
    anchors = _resolve_label_points(geometries, label_points)
    for rid, geom in geometries.items():
        ax.add_patch(PathPatch(geometry_to_path(geom), facecolor="none", **border_kw))
    if label_ids:
        for rid, (x, y) in anchors.items():
            ax.text(x, y, str(rid), **label_kw)


def _autoscale(ax, geometries, margin=0.02):
    """Fit the axes to the union of all region bounding boxes, equal aspect."""
    xs, ys = [], []
    for geom in geometries.values():
        for exterior, _ in _iter_parts(geom):
            pts = np.asarray(exterior, dtype=float)
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    dx = (x.max() - x.min()) * margin
    dy = (y.max() - y.min()) * margin
    ax.set_xlim(x.min() - dx, x.max() + dx)
    ax.set_ylim(y.min() - dy, y.max() + dy)
    ax.set_aspect("equal")


def plot_region_choropleth(
    values,
    geometries,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    norm=None,
    label_points=None,
    label_ids=True,
    title=None,
    cbar_label=None,
    axis_off=True,
    missing_color=MISSING_COLOR,
    border_kw=None,
    label_kw=None,
):
    """
    Fill each region by ``values[region_id]``; draw borders + IDs.

    values      : dict / pandas Series {region_id: scalar}.
    geometries  : dict {region_id: GeoJSON geometry} (data_loader).
    label_points: optional {region_id: (x, y)} guaranteed-inside anchors.
    axis_off    : hide LV95 coordinate axes, leaving just the map + legend.

    Returns (fig, ax, scalar_mappable).
    """
    values = dict(values)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    fig = ax.figure

    finite = [v for v in values.values() if v is not None and np.isfinite(v)]
    if norm is None:
        lo = min(finite) if vmin is None and finite else (vmin or 0.0)
        hi = max(finite) if vmax is None and finite else (vmax or 1.0)
        norm = colors.Normalize(vmin=lo, vmax=hi)
    colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    for rid, geom in geometries.items():
        value = values.get(rid)
        if value is None or not np.isfinite(value):
            face = missing_color
        else:
            face = colormap(norm(value))
        ax.add_patch(
            PathPatch(
                geometry_to_path(geom),
                facecolor=face,
                **{**BORDER_KW, **(border_kw or {})},
            )
        )

    draw_region_borders(
        ax,
        geometries,
        label_points=label_points,
        label_ids=label_ids,
        border_kw=border_kw,
        label_kw=label_kw,
    )
    _autoscale(ax, geometries)

    mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel("Easting LV95")
        ax.set_ylabel("Northing LV95")
    return fig, ax, mappable


# ---------------------------------------------------------------------
# Animation over an ordered sequence of frames
# ---------------------------------------------------------------------


def frames_from_long_df(df, region_col, frame_col, value_col, frame_labels=None):
    """
    Long DataFrame -> ordered [(label, {region_id: value}), ...].

    frame_labels: optional {frame_key: nice_label} (e.g. month -> "January").
    """
    frames = []
    for key, group in df.groupby(frame_col, sort=True):
        mapping = dict(zip(group[region_col].astype(int), group[value_col]))
        label = (frame_labels or {}).get(key, str(key))
        frames.append((label, mapping))
    return frames


def animate_region_choropleth(
    frames,
    geometries,
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    label_points=None,
    label_ids=True,
    title=None,
    cbar_label=None,
    axis_off=True,
    missing_color=MISSING_COLOR,
    interval_ms=700,
    border_kw=None,
    label_kw=None,
):
    """
    Animate a choropleth over ``frames`` (from ``frames_from_long_df``).

    One fixed colour scale is shared across all frames so colours are
    comparable. Returns a matplotlib FuncAnimation (save with ``save_animation``).
    """
    all_values = [
        v for _, m in frames for v in m.values() if v is not None and np.isfinite(v)
    ]
    lo = vmin if vmin is not None else (min(all_values) if all_values else 0.0)
    hi = vmax if vmax is not None else (max(all_values) if all_values else 1.0)
    norm = colors.Normalize(vmin=lo, vmax=hi)
    colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig, ax = plt.subplots(figsize=(11, 8))

    # Build every patch once; recolour them per frame (fast + flicker-free).
    patches = {}
    for rid, geom in geometries.items():
        patch = PathPatch(
            geometry_to_path(geom),
            facecolor=missing_color,
            **{**BORDER_KW, **(border_kw or {})},
        )
        ax.add_patch(patch)
        patches[rid] = patch

    draw_region_borders(
        ax,
        geometries,
        label_points=label_points,
        label_ids=label_ids,
        border_kw=border_kw,
        label_kw=label_kw,
    )
    _autoscale(ax, geometries)

    mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel("Easting LV95")
        ax.set_ylabel("Northing LV95")
    heading = ax.set_title("")

    def update(index):
        label, mapping = frames[index]
        for rid, patch in patches.items():
            value = mapping.get(rid)
            face = (
                colormap(norm(value))
                if value is not None and np.isfinite(value)
                else missing_color
            )
            patch.set_facecolor(face)
        heading.set_text(f"{title} — {label}" if title else str(label))
        return list(patches.values()) + [heading]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, blit=False
    )
    return anim


def save_animation(anim, output_file, *, fps=2, dpi=150):
    """
    Save an animation as .gif (Pillow) or .mp4 (ffmpeg), inferred from suffix.
    """
    output_file = str(output_file)
    if output_file.lower().endswith(".gif"):
        anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(anim._fig)


# ---------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def animate_monthly_regions(
    df,
    geometries,
    *,
    region_col,
    month_col,
    value_col,
    label_points=None,
    output_file=None,
    fps=2,
    **kwargs,
):
    """
    One-call monthly choropleth animation over a long DataFrame.

    Just paste in a table with a region column, a month column (1-12) and a
    value column; months are labelled with their names automatically. Extra
    keyword arguments (cmap, vmin, vmax, title, cbar_label, ...) go straight to
    ``animate_region_choropleth``. Saves the .gif/.mp4 if ``output_file`` is set.
    """
    labels = {i + 1: name for i, name in enumerate(MONTH_NAMES)}
    frames = frames_from_long_df(df, region_col, month_col, value_col, labels)
    anim = animate_region_choropleth(
        frames, geometries, label_points=label_points, **kwargs
    )
    if output_file is not None:
        save_animation(anim, output_file, fps=fps)
    return anim


def plot_region_id_map(
    geometries,
    *,
    ax=None,
    cmap="tab20",
    label_points=None,
    title="Drought regions",
    axis_off=True,
    border_kw=None,
    label_kw=None,
):
    """
    Colour every region with a distinct (cycling) colour and write its ID
    inside — a plain reference map of the region layout. No colour bar, since
    the colours are only there to separate neighbours; the numbers are the key.

    Returns (fig, ax).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    fig = ax.figure

    palette = plt.get_cmap(cmap)
    ids = sorted(geometries)
    for index, rid in enumerate(ids):
        face = palette(index % palette.N)
        ax.add_patch(
            PathPatch(
                geometry_to_path(geometries[rid]),
                facecolor=face,
                **{**BORDER_KW, **(border_kw or {})},
            )
        )

    draw_region_borders(
        ax,
        geometries,
        label_points=label_points,
        label_ids=True,
        border_kw=border_kw,
        label_kw=label_kw,
    )
    _autoscale(ax, geometries)
    if title:
        ax.set_title(title)
    if axis_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel("Easting LV95")
        ax.set_ylabel("Northing LV95")
    return fig, ax
