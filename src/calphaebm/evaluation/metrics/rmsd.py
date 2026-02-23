# src/calphaebm/evaluation/metrics/rmsd.py

"""RMSD calculation with Kabsch alignment."""

from typing import Tuple

import numpy as np


def kabsch_rotate(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal rotation aligning P onto Q after centering.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        (P_aligned, rotation_matrix) where P_aligned = (P - P_com) @ R
    """
    # Center
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Covariance matrix
    C = P_centered.T @ Q_centered

    # SVD
    V, _, Wt = np.linalg.svd(C)

    # Ensure right-handed coordinate system
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    P_aligned = P_centered @ R
    return P_aligned, R


def rmsd_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between two point sets after optimal alignment.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        RMSD value in same units as input.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    if len(P) == 0:
        return 0.0

    P_aligned, _ = kabsch_rotate(P, Q)
    Q_centered = Q - Q.mean(axis=0)

    diff = P_aligned - Q_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def batch_rmsd(trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute RMSD for each frame in a trajectory.

    Args:
        trajectory: (n_frames, N, 3) coordinates.
        reference: (N, 3) reference structure.

    Returns:
        (n_frames,) RMSD values.
    """
    n_frames = trajectory.shape[0]
    rmsds = np.zeros(n_frames)

    for i in range(n_frames):
        rmsds[i] = rmsd_kabsch(trajectory[i], reference)

    return rmsds
