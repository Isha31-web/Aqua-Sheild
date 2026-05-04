"""
exporter.py
-----------
Step 5: Export the pollution map as:
  - result.png  : color-coded heatmap blended over the original image
  - result.tif  : georeferenced GeoTIFF (open in QGIS / ArcGIS)

Color scheme:
  -1 : transparent (nodata / land)
   0 : green  (clean water)
   1 : yellow (low risk)
   2 : orange (moderate)
   3 : red    (high risk)
   4 : purple (critical)
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import RISK_COLORS, RISK_LABELS


def export_png(
    result: np.ndarray,
    path_input: str,
    out_png: str,
) -> None:
    """
    Save a color-coded heatmap overlaid on a false-color base image.

    Parameters
    ----------
    result : (H, W) int8 pollution class array
    path_input    : original EMIT .tif (used to generate the base image)
    out_png       : output PNG file path
    """
    print(f"\n{'='*60}")
    print(f"  STEP 5a: Exporting PNG -> {out_png}")
    print(f"{'='*60}")

    H, W = result.shape

    # Build RGBA overlay from the risk class map
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for val, color in RISK_COLORS.items():
        rgba[result == val] = color

    # Load original image for the base layer (bands 4/3/2 = false color)
    try:
        with rasterio.open(path_input) as src:
            nd = src.nodata
            r  = src.read(min(4, src.count)).astype(np.float32)
            g  = src.read(min(3, src.count)).astype(np.float32)
            b  = src.read(min(2, src.count)).astype(np.float32)

        def _stretch(arr: np.ndarray) -> np.ndarray:
            a = arr.copy()
            if nd is not None:
                a[a == nd] = np.nan
            lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
            return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

        base_rgb = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
        base_rgb = (np.nan_to_num(base_rgb, nan=0.0) * 255).astype(np.uint8)
    except Exception:
        base_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Render: base image + semi-transparent heatmap overlay
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi=150)
    ax.imshow(base_rgb, interpolation="nearest")
    ax.imshow(rgba,     interpolation="nearest", alpha=0.55)
    ax.set_title(
        "Godavari River - Pollution Probability Heatmap\n"
        "(EMIT HSI . CNN Autoencoder + K-Means Clustering)",
        fontsize=13, pad=12,
    )
    ax.axis("off")

    legend_elements = [
        Patch(
            facecolor=np.array(RISK_COLORS[k][:3]) / 255,
            edgecolor="white",
            label=RISK_LABELS[k],
            linewidth=0.5,
        )
        for k in sorted(RISK_LABELS.keys())
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        framealpha=0.85,
        fontsize=9,
        title="Pollution Risk Level",
        title_fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")


def export_geotiff(
    result: np.ndarray,
    meta: dict,
    out_tif: str,
) -> None:
    """
    Save the pollution map as a georeferenced GeoTIFF.

    Class labels and risk descriptions are written as band metadata tags
    so they appear in QGIS / ArcGIS.

    Parameters
    ----------
    result : (H, W) int8 pollution class array
    meta          : metadata dict from loader.py (contains CRS + transform)
    out_tif       : output GeoTIFF file path
    """
    print(f"\n  Exporting GeoTIFF -> {out_tif}")

    profile = {
        "driver":     "GTiff",
        "dtype":      "int8",
        "width":      meta["width"],
        "height":     meta["height"],
        "count":      1,
        "crs":        meta["crs"] or CRS.from_epsg(4326),
        "transform":  meta["transform"],
        "nodata":     -1,
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(result.astype(np.int8), 1)
        dst.update_tags(1, **{
            "CLASS_0": "Clean water",
            "CLASS_1": "Low pollution risk",
            "CLASS_2": "Moderate pollution",
            "CLASS_3": "High pollution",
            "CLASS_4": "Critical / severe",
            "NODATA":  "No data / land / cloud",
            "SOURCE":  "EMIT HSI . CNN Autoencoder + K-Means",
            "RIVER":   "Godavari",
        })

    print(f"  Saved: {out_tif}")
    print(f"\n  Open {out_tif} in QGIS or ArcGIS.")
    print(f"  Color ramp: -1=transparent, 0=green, 1=yellow, 2=orange, 3=red, 4=purple")