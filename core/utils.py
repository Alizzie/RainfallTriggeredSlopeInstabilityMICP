"""Utility functions for the rainfall-slope model."""


def to_lv95(x, y):
    if x < 1_000_000:
        x += 2_000_000
    if y < 1_000_000:
        y += 1_000_000
    return x, y


def get_region_params(x, y, calib, max_snap_m=2000.0):
    """Returns (region_id, drainage_rate, et_rate) for the nearest calibration point."""
    from core import data_loader as dl  # an eure Paketstruktur anpassen

    return dl.get_region_params(x, y, calib, max_snap_m=max_snap_m)
