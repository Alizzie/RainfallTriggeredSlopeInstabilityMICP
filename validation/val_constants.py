"""Constants for the AUC validation scripts"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from core import data_loader as dl

WINDOW_DAYS = 2
SPINUP_DAYS = 120
CONTROLS_PER_EVENT = 3
CONTROL_YEARS = list(range(1991, 2024))
MAX_EVENTS = None  # set e.g. 150 for a quick first run, then None for the full set
CALIB = pd.read_csv(dl.PATH_CALIB)
RNG = np.random.default_rng(0)
OUTDIR = "output/validation"
os.makedirs(OUTDIR, exist_ok=True)


SLOPE_TIF = "data/swissalti_slope/slope_deg_10m.tif"
BETA_MIN, BETA_MAX = 15.0, 45

DEM_TIF = "data/swissalti_slope/swissaltiregio_2056_5728.tif"
RADII_M = [0, 30, 60, 100, 150, 200]

# Soil Cohesion
C_GRID = np.arange(0.0, 12.01, 0.25)  # kPa
