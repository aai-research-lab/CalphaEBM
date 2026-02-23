# src/calphaebm/evaluation/metrics/clash.py

"""Steric clash diagnostics."""

import numpy as np
from typing import Tuple


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances."""
    diff = R[:, None, :] - R[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)


def min_nonbonded(
    R: np.ndarray,
    exclude: int = 2,
) -> Tuple[float, float]:
    """Compute median and minimum nonbonded distances.
    
    Args:
        R: (N, 3) coordinates.
        exclude: Sequence separation cutoff.
        
    Returns:
        (median_distance, min_distance) in Å.
    """
    D = pairwise_distances(R)
    N = D.shape[0]
    
    # Mask out bonded/excluded pairs
    mask = np.ones((N, N), dtype=bool)
    for k in range(-exclude, exclude + 1):
        mask &= ~np.eye(N, k=k, dtype=bool)
    
    iu = np.triu_indices(N, k=1)
    vals = D[iu][mask[iu]]
    
    if len(vals) == 0:
        return float("inf"), float("inf")
    
    return float(np.median(vals)), float(np.min(vals))


def clash_probability(
    trajectory: np.ndarray,
    threshold: float = 3.8,
    exclude: int = 2,
    burnin: int = 0,
) -> Tuple[float, float]:
    """Compute probability of steric clash in trajectory.
    
    Args:
        trajectory: (n_frames, N, 3) coordinates.
        threshold: Minimum allowed distance (Å).
        exclude: Sequence separation cutoff.
        burnin: Number of initial frames to discard.
        
    Returns:
        (p_all, p_post_burnin) clash probabilities.
    """
    n_frames = trajectory.shape[0]
    clashed = np.zeros(n_frames, dtype=bool)
    
    for i in range(n_frames):
        _, min_d = min_nonbonded(trajectory[i], exclude)
        clashed[i] = min_d < threshold
    
    p_all = float(np.mean(clashed))
    
    if burnin < n_frames:
        p_post = float(np.mean(clashed[burnin:]))
    else:
        p_post = 0.0
    
    return p_all, p_post


def batch_min_distances(
    trajectory: np.ndarray,
    exclude: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min and median distances for each frame."""
    n_frames = trajectory.shape[0]
    mins = np.zeros(n_frames)
    medians = np.zeros(n_frames)
    
    for i in range(n_frames):
        med, mn = min_nonbonded(trajectory[i], exclude)
        medians[i] = med
        mins[i] = mn
    
    return mins, medians