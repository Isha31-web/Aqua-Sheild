import numpy as np
import rasterio
from config import EMIT_BANDS

def _safe_band(src, idx: int, nodata) -> np.ndarray:
    band = src.read(int(idx) + 1).astype(np.float32)
    if nodata is not None:
        band[band == nodata] = np.nan
    return band

def _normalize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return arr
    lo, hi = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    return np.clip((arr - lo) / (hi - lo + eps), 0, 1)

def compute_spectral_indices(path: str, meta: dict) -> tuple[dict, np.ndarray]:    
    print(f"\n{'='*60}")
    print(f"  STEP 2: Computing spectral indices")
    print(f"{'='*60}")

    n_bands = meta["count"]
    # Clamp band indices so they never exceed the actual band count
    bi = {k: min(v, n_bands - 1) for k, v in EMIT_BANDS.items()}
    nd = meta["nodata"]

    with rasterio.open(path) as src:
        blue  = _safe_band(src, bi["blue"],  nd)
        green = _safe_band(src, bi["green"], nd)
        red   = _safe_band(src, bi["red"],   nd)
        nir   = _safe_band(src, bi["nir"],   nd)
        swir1 = _safe_band(src, bi["swir1"], nd)

    eps = 1e-6
    ndwi = (green - nir) / (green + nir + eps)
    fai = nir - (red + (swir1 - red) * ((860 - 660) / (1240 - 660)))
    turbidity = red / (green + eps)
    cdom = blue / (red + eps)
    indices = {
        "ndwi":      ndwi,
        "fai":       fai,
        "turbidity": turbidity,
        "cdom":      cdom,
    }
    composite = (
        0.35 * _normalize(fai) +
        0.30 * _normalize(turbidity) +
        0.20 * _normalize(cdom) +
        0.15 * (1 - _normalize(ndwi))                                                                           
    )
    composite[~np.isfinite(composite)] = np.nan

    print(f"  NDWI range      : {np.nanmin(ndwi):.3f} to {np.nanmax(ndwi):.3f}")
    print(f"  FAI range       : {np.nanmin(fai):.3f} to {np.nanmax(fai):.3f}")
    print(f"  Turbidity range : {np.nanmin(turbidity):.3f} to {np.nanmax(turbidity):.3f}")
    print(f"  CDOM range      : {np.nanmin(cdom):.3f} to {np.nanmax(cdom):.3f}")
    print(f"  Composite range : {np.nanmin(composite):.3f} to {np.nanmax(composite):.3f}")

    return indices, composite