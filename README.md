# Aqua-Sheild
# Godavari River HSI Pollution Mapper

Unsupervised CNN-based pollution heatmap from EMIT hyperspectral imagery.

## Project structure

```
pollution_mapper/
    config.py      # All settings: file paths, model params, band mapping
    loader.py      # Step 1: load & inspect the EMIT GeoTIFF
    spectral.py    # Step 2: compute NDWI, FAI, turbidity, CDOM indices
    model.py       # Step 3: patch extraction, CNN autoencoder, embeddings
    cluster.py     # Step 4: K-Means clustering -> pollution risk classes
    exporter.py    # Step 5: save PNG heatmap + georeferenced GeoTIFF
    main.py        # Entry point — wires all modules together
    README.md      # This file
```

## Setup

Install dependencies:
```bash
pip install rasterio numpy matplotlib scikit-learn torch torchvision tqdm scipy pillow
```

## Usage

1. Open `config.py` and set `INPUT_TIF` to the path of your EMIT .tif file.
2. Optionally adjust `EPOCHS`, `N_CLUSTERS`, `PATCH_SIZE` etc. in `config.py`.
3. Run:

```bash
python main.py
```

## Outputs

| File               | Description                                          |
|--------------------|------------------------------------------------------|
| `pollution_map.png`| Color-coded heatmap overlaid on false-color image    |
| `pollution_map.tif`| Georeferenced GeoTIFF, openable in QGIS / ArcGIS     |

## Pollution risk classes

| Value | Color  | Label              |
|-------|--------|--------------------|
| -1    | None   | No data / land     |
|  0    | Green  | Clean water        |
|  1    | Yellow | Low pollution risk |
|  2    | Orange | Moderate pollution |
|  3    | Red    | High pollution     |
|  4    | Purple | Critical / severe  |

## Pipeline overview

```
EMIT .tif (285 bands)
    |
    v
[loader.py]     Read metadata, no full load into memory
    |
    v
[spectral.py]   Compute NDWI, FAI, turbidity, CDOM -> composite score
    |
    v
[model.py]      Extract 7x7 patches -> train CNN autoencoder -> 64-dim embeddings
    |
    v
[cluster.py]    K-Means on embeddings, rank by composite -> pollution_map
    |
    v
[exporter.py]   Save PNG overlay + GeoTIFF
```
