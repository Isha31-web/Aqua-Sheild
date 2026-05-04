"""
loader.py
---------
Step 1: Load and inspect the EMIT GeoTIFF.
Reads metadata without loading all bands into memory (safe for 4GB+ files).
"""

import os
import sys
import numpy as np
import rasterio


def load_and_inspect(path: str) -> dict:
    """
    Open the GeoTIFF, print a summary, and return a metadata dict.

    Parameters
    ----------
    path : str
        Path to the EMIT .tif file.

    Returns
    -------
    dict with keys: driver, count, dtype, crs, transform, width, height, nodata
    """
    print(f"\n{'='*60}")
    print(f"  STEP 1: Loading {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        print("Please update INPUT_TIF in config.py with the correct path.")
        sys.exit(1)

    with rasterio.open(path) as src:
        meta = {
            "driver":    src.driver,
            "count":     src.count,
            "dtype":     src.dtypes[0],
            "crs":       src.crs,
            "transform": src.transform,
            "width":     src.width,
            "height":    src.height,
            "nodata":    src.nodata,
        }
        size_gb = (
            src.count * src.width * src.height
            * np.dtype(src.dtypes[0]).itemsize
        ) / 1e9

        print(f"  Bands      : {src.count}")
        print(f"  Dimensions : {src.width} x {src.height} px")
        print(f"  Data type  : {src.dtypes[0]}")
        print(f"  CRS        : {src.crs}")
        print(f"  NoData val : {src.nodata}")
        print(f"  Transform  : {src.transform}")
        print(f"  Est. size  : {size_gb:.2f} GB")

    return meta