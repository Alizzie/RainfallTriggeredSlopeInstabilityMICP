import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

values = [0, 1, 2, 3]  # the classes in your maps
n = len(values)
target_cells_part = 50000000  # ~50M cells per strip


def compare(path1, path2):
    with rasterio.open(path1) as r1, rasterio.open(path2) as r2:
        if (r1.width, r1.height) != (r2.width, r2.height):
            sys.exit(f"Rasters have different sizes: {r1.width}x{r1.height} vs {r2.width}x{r2.height}. "
                     "Align them first (e.g. QGIS 'Align Rasters' or gdalwarp).")
        if r1.crs != r2.crs:
            sys.exit(f"Different CRS: {r1.crs} vs {r2.crs}. Reproject one of them first.")
        if not r1.transform.almost_equals(r2.transform):
            sys.exit("Same size but different extent/pixel grid. Align them first.")

        nodata1, nodata2 = r1.nodata, r2.nodata
        width, height = r1.width, r1.height
        rows_per_part = max(1, target_cells_part // width)

        counts = np.zeros(n * n, dtype=np.int64)  # flattened matrix
        skipped = 0  # nodata / out of range cells

        for row_off in range(0, height, rows_per_part):
            h = min(rows_per_part, height - row_off)
            win = Window(0, row_off, width, h)
            a = r1.read(1, window=win).ravel()
            b = r2.read(1, window=win).ravel()

            # valid = in 0...3 in both maps and not nodata in either
            valid = (a >= 0) & (a < n) & (b >= 0) & (b < n)
            if nodata1 is not None:
                valid &= a != nodata1
            if nodata2 is not None:
                valid &= b != nodata2
            skipped += valid.size - np.count_nonzero(valid)

            a = a[valid].astype(np.int64)
            b = b[valid].astype(np.int64)
            # each (x, y) pair -> single index x*4 + y  then count all at once
            counts += np.bincount(a * n + b, minlength=n * n)

            print(f"\r  processed {min(row_off + h, height):,}/{height:,} rows", end="", flush=True)
        print()

    matrix = counts.reshape(n, n)
    total = matrix.sum()
    if total == 0:
        sys.exit("no valid overlapping cells found (check nodata settings).")

    # Overall agreement
    agree = np.trace(matrix)
    print(f"\nValid cells compared:  {total:,}")
    print(f"Skipped (nodata/other): {skipped:,}")
    print(f"\noverall agreement: {agree / total * 100:.2f}%  ({agree:,} cells same value)")

    # cross overlap table
    labels_x = [f"map1={v}" for v in values]
    labels_y = [f"map2={v}" for v in values]
    df_counts = pd.DataFrame(matrix, index=labels_x, columns=labels_y)
    df_pct_total = df_counts / total * 100
    # row % of the cells that are x in map 1, what share is y in map 2
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_pct_row = pd.DataFrame(np.where(row_sums > 0, matrix / row_sums * 100, np.nan),
                                  index=labels_x, columns=labels_y)

    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n% of all valid cells:")
    print(df_pct_total.to_string())
    print("\n% within each map1 class:")
    print(df_pct_row.to_string())

    # per class agreement (how much of each map1 class kept its value)
    print("\nper class agreement:")
    for i, v in enumerate(values):
        if row_sums[i, 0]:
            print(f"value {v}: {matrix[i, i] / row_sums[i, 0] * 100:.2f}% overlap "
                  f"({row_sums[i, 0]:,} cells in map1)")




compare('try6.tif', 'micp_landslide_categories_1.0.tif')