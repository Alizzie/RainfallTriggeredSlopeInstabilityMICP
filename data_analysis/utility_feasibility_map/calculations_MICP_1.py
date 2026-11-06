import rasterio
import numpy as np

counts = {0: 0, 1: 0, 2: 0, 3: 0}

with rasterio.open("try6.tif") as src:
    for _, window in src.block_windows(1):
        data = src.read(1, window=window)
        vals, cnts = np.unique(data, return_counts=True)
        for v, c in zip(vals, cnts):
            if v in counts:
                counts[v] += int(c)

pixelsum = sum(counts.values())
counts_percentage = {k: v/pixelsum for k, v in counts.items()}
print(f"\n0: n LS, n MICP: {counts_percentage[0]:.2%}"
      f"\n1: y LS, n MICP: {counts_percentage[1]:.2%}"
      f"\n2: n LS, y MICP: {counts_percentage[2]:.2%}"
      f"\n3: y LS, y MICP: {counts_percentage[3]:.2%}")

prop = counts[3] / (counts[3] + counts[1])
print(f"\nMICP coverable share of landslide affected area: {prop:.2%}.")

MICP_CH = counts_percentage[2] + counts_percentage[3]
print(f"\nMICP coverable share of Switzerland: {MICP_CH:.2%}.\n")

# print(counts)


