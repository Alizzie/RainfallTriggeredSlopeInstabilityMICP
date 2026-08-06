import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import constants as const, data_loader as dl, physics
from core import val_metrics as vm

inv = dl.load_wsl_inventory()  # full, unfiltered
print((inv["date"].dt.year < 1993).sum(), "events before 1993")

# Get rows
for _, row in inv.iterrows():
    rain = dl.load_rainfall(row["x"], row["y"], range(1991, 1993))
    print("zero-rainfall days:", (rain == 0).mean())  # vs a mid-record year
