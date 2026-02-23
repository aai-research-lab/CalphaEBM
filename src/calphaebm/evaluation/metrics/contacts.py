# src/calphaebm/evaluation/metrics/contacts.py

"""Native contact analysis (Q) and contact counting."""

import numpy as np
from typing import Tuple, Optional


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances."""
    diff = R[:, None, :] - R[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)


def native_contact_set(
    R_ref: np.ndarray,
    cutoff: float = 8.0,
    exclude: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build native contact set from reference structure.
    
    Args:
        R_ref: (N, 3) reference coordinates.
        cutoff: Distance cutoff for defining contacts (Å).
        exclude: Sequence separation cutoff (|i-j| <= exclude excluded).
        
    Returns:
        (i_indices, j_indices, d0_distances) for native contacts.
    """
    D = pairwise_distances(R_ref)
    N = D.shape[0]
    
    # Sequence separation mask
    sep = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    mask = sep > exclude
    
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    good = mask[iu] & (D[iu] < cutoff)
    
    i = iu[0][good].astype(np.int64)
    j = iu[1][good].astype(np.int64)
    d0 = D[i, j].astype(np.float64)
    
    return i, j, d0


def q_hard(
    R: np.ndarray,
    native_i: np.ndarray,
    native_j: np.ndarray,
    cutoff: float = 8.0,
) -> float:
    """Compute Q_hard: fraction of native contacts present with hard cutoff.
    
    Args:
        R: (N, 3) coordinates.
        native_i, native_j: Native contact indices.
        cutoff: Distance cutoff for considering contact present.
        
    Returns:
        Q value in [0, 1].
    """
    if native_i.size == 0:
        return 0.0
    
    rij = R[native_i] - R[native_j]
    dij = np.sqrt(np.sum(rij * rij, axis=1) + 1e-12)
    
    return float(np.mean(dij < cutoff))


def q_smooth(
    R: np.ndarray,
    native_i: np.ndarray,
    native_j: np.ndarray,
    d0: np.ndarray,
    beta: float = 5.0,
    lam: float = 1.8,
) -> float:
    """Compute Q_smooth: Best-style smooth contact fraction.
    
    s_ij = 1 / (1 + exp(beta * (d_ij - lam * d0_ij)))
    Q = mean(s_ij)
    
    Args:
        R: (N, 3) coordinates.
        native_i, native_j: Native contact indices.
        d0: Native contact distances.
        beta: Sharpness parameter (1/Å).
        lam: Tolerance factor multiplying native distance.
        
    Returns:
        Q value in [0, 1].
    """
    if native_i.size == 0:
        return 0.0
    
    rij = R[native_i] - R[native_j]
    dij = np.sqrt(np.sum(rij * rij, axis=1) + 1e-12)
    
    x = beta * (dij - lam * d0)
    x = np.clip(x, -60.0, 60.0)  # Prevent overflow
    s = 1.0 / (1.0 + np.exp(x))
    
    return float(np.mean(s))


def contact_count(
    R: np.ndarray,
    cutoff: float = 8.0,
    exclude: int = 2,
) -> int:
    """Count number of contacts (pairs within cutoff, excluding local)."""
    D = pairwise_distances(R)
    N = D.shape[0]
    
    sep = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    mask = sep > exclude
    
    iu = np.triu_indices(N, k=1)
    good = mask[iu] & (D[iu] < cutoff)
    
    return int(good.sum())