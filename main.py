"""
main.py
-------
Entry point for the Godavari River HSI Pollution Mapper.

Run with:
    python main.py

Edit config.py to change file paths and model parameters before running.

Pipeline:
    Step 1  loader.py    : load & inspect the EMIT GeoTIFF
    Step 2  spectral.py  : compute NDWI, FAI, turbidity, CDOM + composite score
    Step 3  model.py     : extract patches, train CNN autoencoder, get embeddings
    Step 4  cluster.py   : cluster embeddings -> ranked pollution classes
    Step 5  exporter.py  : save result.png + result.tif
"""

from config   import INPUT_TIF, OUTPUT_PNG, OUTPUT_TIF, MAX_PATCHES, DEVICE, RISK_LABELS
from loader   import load_and_inspect
from spectral import compute_spectral_indices
from model    import extract_patches, train_autoencoder, extract_embeddings
from cluster  import cluster_embeddings
from exporter import export_png, export_geotiff


def main() -> None:
    print("\n" + "=" * 60)
    print("  Godavari River Pollution Mapper")
    print("  EMIT HSI . CNN Autoencoder Pipeline")
    print("=" * 60)
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Input  : {INPUT_TIF}")

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    meta = load_and_inspect(INPUT_TIF)

    # ------------------------------------------------------------------
    # Step 2: Spectral indices
    # ------------------------------------------------------------------
    indices, composite = compute_spectral_indices(INPUT_TIF, meta)

    # ------------------------------------------------------------------
    # Step 3: CNN Autoencoder
    # ------------------------------------------------------------------
    patches, centers, n_ch = extract_patches(
        INPUT_TIF, meta, composite, indices, MAX_PATCHES
    )
    model      = train_autoencoder(patches, n_ch)
    embeddings = extract_embeddings(model, patches)

    # ------------------------------------------------------------------
    # Step 4: Clustering
    # ------------------------------------------------------------------
    result = cluster_embeddings(embeddings, composite, centers, meta)

    # ------------------------------------------------------------------
    # Step 5: Export
    # ------------------------------------------------------------------
    export_png(result, INPUT_TIF, OUTPUT_PNG)
    export_geotiff(result, meta, OUTPUT_TIF)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_valid = (result >= 0).sum()
    print(f"\n{'='*60}")
    print(f"  DONE! Pixel summary:")
    print(f"{'='*60}")
    for k, label in RISK_LABELS.items():
        count = (result == k).sum()
        pct   = 100 * count / max(total_valid, 1)
        print(f"  {label:<30} {count:>10,} px  ({pct:.1f}%)")
    print(f"\n  Outputs saved:")
    print(f"    {OUTPUT_PNG}")
    print(f"    {OUTPUT_TIF}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()