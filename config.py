"""
config.py
---------
All tunable parameters and paths for the Godavari pollution mapper.
Edit this file before running main.py.
"""

import sys
import torch

# Force UTF-8 output on Windows (avoids UnicodeEncodeError in cmd/PowerShell)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# File paths
# =============================================================================

INPUT_TIF  = "emit_final.tif"    # Path to your EMIT .tif file
OUTPUT_PNG = "result.png" # Output heatmap overlay
OUTPUT_TIF = "result.tif" # Output georeferenced GeoTIFF

# =============================================================================
# Model & training parameters
# =============================================================================

PATCH_SIZE  = 7        # Spatial window size fed to CNN (pixels)
STRIDE      = 4        # Extraction stride (lower = denser map, slower)
MAX_PATCHES = 80_000   # Max patches sampled for training (RAM-safe limit)
N_CLUSTERS  = 5        # Number of pollution risk classes
BATCH_SIZE  = 512      # Training batch size
EPOCHS      = 30       # Autoencoder training epochs
LATENT_DIM  = 64       # Size of compressed embedding vector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# EMIT band mapping
# Approximate band indices for key wavelengths.
# EMIT covers ~380-2500nm across 285 bands.
# Adjust these if you have the official wavelength metadata.
# =============================================================================

EMIT_BANDS = {
    "blue":  10,   # ~450nm
    "green": 25,   # ~550nm
    "red":   35,   # ~660nm
    "nir":   60,   # ~860nm
    "swir1": 120,  # ~1240nm
    "swir2": 180,  # ~1640nm
}

# =============================================================================
# Pollution risk class colors (RGBA) and labels
# Used in both PNG export and GeoTIFF metadata
# =============================================================================

RISK_COLORS = {
    -1: (0,   0,   0,   0),    # Transparent / nodata
     0: (39,  174, 96,  220),  # Clean          - green
     1: (241, 196, 15,  220),  # Low risk       - yellow
     2: (230, 126, 34,  220),  # Moderate risk  - orange
     3: (192, 57,  43,  220),  # High risk      - red
     4: (109, 33,  79,  220),  # Critical       - dark purple
}

RISK_LABELS = {
    0: "Clean water",
    1: "Low pollution risk",
    2: "Moderate pollution",
    3: "High pollution",
    4: "Critical / severe",
}