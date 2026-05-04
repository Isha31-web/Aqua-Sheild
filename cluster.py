"""
cluster.py
----------
Step 4: Cluster CNN embeddings into ranked pollution risk classes.

Uses MiniBatchKMeans for speed on large embedding sets.
Clusters are automatically ranked from cleanest (0) to most polluted (4)
by comparing their mean composite score from the spectral indices.
Gap pixels (not covered by any patch center) are filled via
nearest-neighbor interpolation using scipy's distance_transform_edt.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from config import N_CLUSTERS


def cluster_embeddings(
    embeddings: np.ndarray,
    composite: np.ndarray,
    centers: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Group patch embeddings into N_CLUSTERS classes and rank them by
    pollution level using the composite spectral score as a guide.

    Parameters
    ----------
    embeddings : (N, latent_dim) array of CNN encoder outputs
    composite  : (H, W) pollution composite score from spectral.py
    centers    : (N, 2) array of (row, col) patch center coordinates
    meta       : metadata dict from loader.py

    Returns
    -------
    result : (H, W) int8 array
        Values: -1 = nodata, 0 = clean, ..., N_CLUSTERS-1 = most polluted
    """
    print(f"\n{'='*60}")
    print(f"  STEP 4: Clustering into {N_CLUSTERS} pollution classes")
    print(f"{'='*60}")

    H, W = meta["height"], meta["width"]

    # ------------------------------------------------------------------
    # 1. Normalize embeddings then cluster with MiniBatchKMeans
    # ------------------------------------------------------------------
    scaler   = StandardScaler()
    emb_norm = scaler.fit_transform(embeddings)

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=4096,
        n_init=10,
        random_state=42,
        verbose=0,
    )
    labels = kmeans.fit_predict(emb_norm)

    sizes = {i: int((labels == i).sum()) for i in range(N_CLUSTERS)}
    print(f"  Cluster sizes: {sizes}")

    # ------------------------------------------------------------------
    # 2. Rank clusters by their mean composite score
    #    Rank 0 = cleanest, Rank (N_CLUSTERS-1) = most polluted
    # ------------------------------------------------------------------
    cluster_scores: dict[int, float] = {}
    for c in range(N_CLUSTERS):
        mask      = labels == c
        cy, cx    = centers[mask, 0], centers[mask, 1]
        scores    = composite[cy, cx]
        cluster_scores[c] = float(np.nanmean(scores))

    sorted_clusters = sorted(cluster_scores, key=lambda c: cluster_scores[c])
    rank_map        = {old: new for new, old in enumerate(sorted_clusters)}
    ranked_labels   = np.array([rank_map[l] for l in labels])

    print(f"  Cluster -> pollution rank mapping:")
    for old, score in cluster_scores.items():
        print(f"    Cluster {old} -> rank {rank_map[old]}  (mean composite={score:.3f})")

    # ------------------------------------------------------------------
    # 3. Paint patch centers onto full-resolution map
    # ------------------------------------------------------------------
    result = np.full((H, W), -1, dtype=np.int8)
    for i, (cy, cx) in enumerate(centers):
        result[cy, cx] = ranked_labels[i]

    # ------------------------------------------------------------------
    # 4. Fill gaps between patch centers using nearest-neighbor EDT
    # ------------------------------------------------------------------
    unknown = result == -1
    _, nearest_idx = distance_transform_edt(unknown, return_indices=True)
    result  = result[nearest_idx[0], nearest_idx[1]].astype(np.int8)

    # ------------------------------------------------------------------
    # 5. Re-apply nodata mask (land, clouds, no-data stripes)
    # ------------------------------------------------------------------
    nodata_mask = ~np.isfinite(composite)
    result[nodata_mask] = -1

    return result