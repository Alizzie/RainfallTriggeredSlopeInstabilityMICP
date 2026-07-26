"""Utility functions for the rainfall-slope model."""


def to_lv95(x, y):
    """Convert coordinates to LV95 (EPSG:2056) if they are in LV03 (EPSG:21781)."""
    if x < 1_000_000:
        x += 2_000_000
    if y < 1_000_000:
        y += 1_000_000
    return x, y
