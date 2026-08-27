"""
runout_model.py - Core, map-agnostic landslide runout logic.

Everything here operates on plain NumPy arrays (DEM + cell size), with no
hardcoded paths or canton assumptions, so the same functions work for Glarus
today and any other canton later just by feeding in different rasters.

ASSUMES a north-up raster (row 0 = north, increasing row = south), the normal
GeoTIFF convention. Compass directions map to array steps as:
    N: (row-1, col)     S: (row+1, col)
    E: (row,   col+1)   W: (row,   col-1)

MODEL LOGIC (as specified):
  1. Direction selection: look at the DEM at the 4 immediate neighbors of a
     release point. Neighbors LOWER than the release point are candidate
     runout directions.
       - exactly 1 candidate  -> use it directly (clear downhill side)
       - >1 candidates (incl. a true 4-way peak) -> simulate ALL candidates,
         keep whichever produces the WORST damage (see damage_profile.py)
       - 0 candidates -> point sits in a local pit; flagged, skipped by caller
  2. Runout propagation: walk outward one cell at a time in the chosen
     direction. Stop when:
       - local slope drops below FLAT_SLOPE_DEG (reached a flat valley floor)
       - elevation rises again by more than UPHILL_TOLERANCE_M (hit the next
         maxima / far side of a ridge)
       - MAX_RADIUS_M is reached (safety cap)
  3. Swath: the 1-cell-wide path is buffered by SWATH_WIDTH_M perpendicular
     to the direction of travel, turning it into a damage corridor.
"""

import numpy as np

DIRECTIONS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


def flow_directions_at(dem, row, col):
    """Candidate downhill directions from one point, by comparing to its 4
    immediate N/E/S/W neighbors. Returns a list of direction strings.

    Empty list -> point is a local pit (no lower neighbor at all).
    One entry  -> single clear downhill side.
    2-4 entries -> ambiguous / peak-like; caller should try all and pick worst.
    """
    h, w = dem.shape
    z0 = dem[row, col]
    candidates = []
    for d, (dr, dc) in DIRECTIONS.items():
        r, c = row + dr, col + dc
        if 0 <= r < h and 0 <= c < w and np.isfinite(dem[r, c]):
            if dem[r, c] < z0:
                candidates.append(d)
    return candidates


def simulate_runout_path(
    dem,
    start_row,
    start_col,
    direction,
    cell_size,
    max_radius_m,
    flat_slope_deg=5.0,
    uphill_tolerance_m=0.5,
):
    """Walk from (start_row, start_col) in one compass direction until a stop
    condition is hit. Returns (path_cells, distance_m, stop_reason).

    path_cells: list of (row, col), INCLUDING the start cell.
    stop_reason: one of "flat", "uphill", "max_radius", "edge".
    """
    dr, dc = DIRECTIONS[direction]
    h, w = dem.shape

    path = [(start_row, start_col)]
    prev_z = dem[start_row, start_col]
    distance = 0.0
    max_steps = int(max_radius_m / cell_size) + 1

    r, c = start_row, start_col
    for _ in range(max_steps):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or not np.isfinite(dem[nr, nc]):
            return path, distance, "edge"

        z = dem[nr, nc]
        rise = z - prev_z
        slope_deg = np.degrees(np.arctan2(abs(rise), cell_size))

        if rise > uphill_tolerance_m:
            # started climbing again -> hit the next rise/ridge
            return path, distance, "uphill"

        distance += cell_size
        path.append((nr, nc))

        if slope_deg < flat_slope_deg:
            # reached a flat valley floor - one more cell then stop
            return path, distance, "flat"

        if distance >= max_radius_m:
            return path, distance, "max_radius"

        r, c, prev_z = nr, nc, z

    return path, distance, "max_radius"


def build_swath_mask(
    path_cells,
    direction,
    toe_width_m,
    cell_size,
    raster_shape,
    head_width_m=None,
    shape_mode="fan",
):
    """Turn a straight N/E/S/W path into a boolean damage-corridor mask.

    shape_mode:
      "fan"       - width grows linearly from head_width_m (at the release
                    point) to toe_width_m (at the far end of the runout) -
                    more physically representative of how debris/rockslide
                    material spreads laterally as it loses lateral
                    confinement moving downslope. THIS IS THE DEFAULT.
      "rectangle" - constant width = toe_width_m along the whole path (the
                    original, simpler behavior).

    head_width_m defaults to toe_width_m / 4 if not given. Ignored when
    shape_mode="rectangle".

    WINDOWED: allocates only a small local array sized to the path's bounding
    box + toe_width_m (the widest point), not the full national grid. Returns
    (local_mask, row_offset, col_offset) - stitch into a full-size array via
    paste_local_mask().

    N/S travel -> width expands east-west (columns).
    E/W travel -> width expands north-south (rows).
    """
    if head_width_m is None:
        head_width_m = toe_width_m / 4.0

    max_half_width_px = max(int(round((toe_width_m / 2.0) / cell_size)), 0)
    h, w = raster_shape
    vertical = direction in ("N", "S")

    rows = [r for r, c in path_cells]
    cols = [c for r, c in path_cells]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)

    # bounding box padded by the WIDEST point (toe_width_m) - correct upper
    # bound regardless of shape_mode, since fan mode never exceeds toe_width_m
    if vertical:
        c_min -= max_half_width_px
        c_max += max_half_width_px
    else:
        r_min -= max_half_width_px
        r_max += max_half_width_px

    r_min, c_min = max(r_min, 0), max(c_min, 0)
    r_max, c_max = min(r_max, h - 1), min(c_max, w - 1)

    local_h, local_w = r_max - r_min + 1, c_max - c_min + 1
    local_mask = np.zeros((local_h, local_w), dtype=bool)

    n_steps = len(path_cells)
    for i, (r, c) in enumerate(path_cells):
        if shape_mode == "fan" and n_steps > 1:
            frac = i / (n_steps - 1)
            local_width_m = head_width_m + frac * (toe_width_m - head_width_m)
        else:
            local_width_m = toe_width_m
        half_width_px = max(int(round((local_width_m / 2.0) / cell_size)), 0)

        lr, lc = r - r_min, c - c_min
        if vertical:
            lc0 = max(lc - half_width_px, 0)
            lc1 = min(lc + half_width_px + 1, local_w)
            local_mask[lr, lc0:lc1] = True
        else:
            lr0 = max(lr - half_width_px, 0)
            lr1 = min(lr + half_width_px + 1, local_h)
            local_mask[lr0:lr1, lc] = True

    return local_mask, r_min, c_min


def paste_local_mask(full_mask, local_mask, row_offset, col_offset):
    """Stitch a windowed mask (from build_swath_mask) into a full-size array."""
    h, w = local_mask.shape
    full_mask[row_offset : row_offset + h, col_offset : col_offset + w] |= local_mask


def rasterize_vector_to_grid(
    vector_path, ref_shape, ref_transform, id_field=None, dtype=np.int32
):
    """Convenience helper: rasterize a vector layer (roads, railway, building
    footprints) onto the SAME grid as your DEM/FoS rasters.

    If id_field is given, burns that field's value per feature (e.g. a unique
    building ID, so distinct buildings can be counted after intersecting a
    swath - not just "building present/absent"). Otherwise burns a constant 1
    (plain binary presence mask).

    Only imports geopandas/rasterio.features lazily, since not every use of
    this module needs vector I/O.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(vector_path)
    if id_field:
        shapes = [(geom, int(val)) for geom, val in zip(gdf.geometry, gdf[id_field])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]

    return rasterize(
        shapes=shapes, out_shape=ref_shape, transform=ref_transform, fill=0, dtype=dtype
    )
