"""
model.py
--------
Step 3: CNN Autoencoder pipeline.

Contains:
  - PatchDataset      : PyTorch Dataset wrapping extracted patches
  - ConvAutoencoder   : Encoder-decoder CNN architecture
  - extract_patches() : Slide a window across the image to build training data
  - train_autoencoder(): Train the autoencoder to compress spectral patches
  - extract_embeddings(): Run the trained encoder to get per-patch embeddings
"""

import os
import copy
import tempfile

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    PATCH_SIZE, STRIDE, MAX_PATCHES,
    BATCH_SIZE, EPOCHS, LATENT_DIM,
    DEVICE, EMIT_BANDS,
)


# =============================================================================
# Dataset
# =============================================================================

class PatchDataset(Dataset):
    """Wraps a numpy patch array as a PyTorch Dataset."""

    def __init__(self, patches: np.ndarray):
        self.patches = torch.tensor(patches, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.patches[idx]


# =============================================================================
# Model architecture
# =============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for hyperspectral patches.

    Encoder: 2x Conv2d -> AdaptiveAvgPool -> Linear -> latent vector
    Decoder: Linear -> reshape -> 2x Conv2d -> reconstructed patch
    """

    def __init__(self, in_channels: int, patch_size: int, latent_dim: int):
        super().__init__()

        self.patch_size = patch_size

        # Encoder: spatial + spectral compression
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # -> (B, 128, 1, 1)
        )
        self.fc_enc = nn.Linear(128, latent_dim)

        # Decoder: expand back to original patch dimensions
        self.fc_dec = nn.Linear(latent_dim, 128 * patch_size * patch_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 128, self.patch_size, self.patch_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# =============================================================================
# Patch extraction
# =============================================================================

def _normalize_band(arr: np.ndarray, nodata) -> np.ndarray:
    """Per-band 2-98 percentile stretch, nodata -> 0."""
    a = arr.copy()
    if nodata is not None:
        a[a == nodata] = np.nan
    lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(a, nan=0.0).astype(np.float32)


def _normalize_index(arr: np.ndarray) -> np.ndarray:
    """Normalize a spectral index array to [0, 1]."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    out = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def extract_patches(
    path: str,
    meta: dict,
    composite: np.ndarray,
    indices: dict,
    max_patches: int = MAX_PATCHES,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Slide a PATCH_SIZE x PATCH_SIZE window across the image and build a
    training array of spectral patches.

    Channels = 26 sampled spectral bands + 4 spectral indices (NDWI, FAI,
    turbidity, CDOM) = 30 total.

    Parameters
    ----------
    path        : path to the EMIT .tif
    meta        : metadata dict from loader
    composite   : (H, W) composite pollution score (used as valid-pixel mask)
    indices     : dict of spectral index arrays from spectral.py
    max_patches : maximum number of patches to extract

    Returns
    -------
    patches : (N, C, patch_size, patch_size) float32 array
    centers : (N, 2) int array of (row, col) center coordinates
    n_ch    : total number of channels C
    """
    print(f"\n{'='*60}")
    print(f"  STEP 3a: Extracting patches (max {max_patches:,})")
    print(f"{'='*60}")

    H, W    = meta["height"], meta["width"]
    n_bands = meta["count"]
    half    = PATCH_SIZE // 2

    # Sample band indices evenly across the full spectrum + water-sensitive bands
    # Cast to plain Python int — rasterio rejects numpy.int64
    sample_bands = sorted(set(
        [int(x) for x in np.linspace(0, n_bands - 1, 20, dtype=int)] +
        [int(min(v, n_bands - 1)) for v in EMIT_BANDS.values()]
    ))
    n_spectral = len(sample_bands)
    n_ch = n_spectral + 4   # +4 for the spectral indices

    print(f"  Using {n_spectral} spectral bands + 4 indices = {n_ch} total channels")

    # Build valid-pixel center grid
    valid_mask = np.isfinite(composite)
    ys = np.arange(half, H - half, STRIDE)
    xs = np.arange(half, W - half, STRIDE)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()

    mask = valid_mask[yy, xx]
    yy, xx = yy[mask], xx[mask]

    if len(yy) > max_patches:
        sel = np.random.choice(len(yy), max_patches, replace=False)
        yy, xx = yy[sel], xx[sel]

    centers = np.stack([yy, xx], axis=1)
    print(f"  Extracting {len(centers):,} patches of size {PATCH_SIZE}x{PATCH_SIZE}x{n_ch}...")

    # Read and normalize spectral bands
    nd = meta["nodata"]
    band_arrays: list[np.ndarray] = []
    with rasterio.open(path) as src:
        for bi in tqdm(sample_bands, desc="  Reading bands"):
            raw = src.read(int(bi) + 1).astype(np.float32)
            band_arrays.append(_normalize_band(raw, nd))

    # Append the 4 spectral index channels
    all_channels: list[np.ndarray] = band_arrays + [
        _normalize_index(indices["ndwi"]),
        _normalize_index(indices["fai"]),
        _normalize_index(indices["turbidity"]),
        _normalize_index(indices["cdom"]),
    ]

    # Extract patches
    patches = np.zeros((len(centers), n_ch, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i, (cy, cx) in enumerate(tqdm(centers, desc="  Building patches")):
        for c, arr in enumerate(all_channels):
            patches[i, c] = arr[cy - half:cy + half + 1, cx - half:cx + half + 1]

    print(f"  Patches shape: {patches.shape}")
    return patches, centers, n_ch


# =============================================================================
# Training
# =============================================================================

def train_autoencoder(patches: np.ndarray, n_channels: int) -> ConvAutoencoder:
    """
    Train the CNN autoencoder on the extracted patches.

    Uses CosineAnnealingLR scheduler and saves the best checkpoint
    to the system temp directory (cross-platform, works on Windows).

    Parameters
    ----------
    patches    : (N, C, H, W) float32 patch array
    n_channels : number of input channels C

    Returns
    -------
    Trained ConvAutoencoder with best weights loaded.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 3b: Training CNN Autoencoder on {DEVICE.upper()}")
    print(f"{'='*60}")

    dataset   = PatchDataset(patches)
    loader    = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model     = ConvAutoencoder(n_channels, PATCH_SIZE, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.MSELoss()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Cross-platform temp checkpoint (works on Windows + Linux/Mac)
    tmp_ckpt   = os.path.join(tempfile.gettempdir(), "_godavari_best_model.pt")
    best_loss  = float("inf")
    best_state = None   # in-memory fallback if disk save fails

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss: 
            best_loss = avg_loss
            try:
                torch.save(model.state_dict(), tmp_ckpt)
            except Exception:
                best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.5f}  lr={lr:.6f}")

    # Restore best weights
    if os.path.exists(tmp_ckpt):
        model.load_state_dict(torch.load(tmp_ckpt, map_location=DEVICE))
        try:
            os.remove(tmp_ckpt)
        except Exception:
            pass
    elif best_state is not None:  
        model.load_state_dict(best_state)

    print(f"  Best loss: {best_loss:.5f}")
    return model


# =============================================================================
# Embedding extraction
# =============================================================================

def extract_embeddings(model: ConvAutoencoder, patches: np.ndarray) -> np.ndarray:
    """
    Run the trained encoder over all patches and collect the latent vectors.

    Parameters
    ----------
    model   : trained ConvAutoencoder
    patches : (N, C, H, W) patch array

    Returns
    -------
    embeddings : (N, LATENT_DIM) float32 numpy array
    """
    print(f"\n  Extracting embeddings...")
    model.eval()

    dataset = PatchDataset(patches)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Encoding"):
            batch = batch.to(DEVICE)
            _, z  = model(batch)
            all_embeddings.append(z.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)