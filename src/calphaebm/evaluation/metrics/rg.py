# src/calphaebm/evaluation/metrics/rg.py

"""Radius of gyration calculation."""

import numpy as np


def radius_of_gyration(R: np.ndarray) -> float:
    """Compute radius of gyration.

    Rg = sqrt(mean(||r_i - r_com||²))

    Args:
        R: (N, 3) coordinates.

    Returns:
        Radius of gyration in same units as input.
    """
    com = R.mean(axis=0)
    return float(np.sqrt(((R - com) ** 2).sum(axis=1).mean()))


def batch_rg(trajectory: np.ndarray) -> np.ndarray:
    """Compute Rg for each frame in trajectory."""
    n_frames = trajectory.shape[0]
    rgs = np.zeros(n_frames)

    for i in range(n_frames):
        rgs[i] = radius_of_gyration(trajectory[i])

    return rgs


def delta_rg(trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute ΔRg = Rg(t) - Rg(ref)."""
    rg_ref = radius_of_gyration(reference)
    rgs = batch_rg(trajectory)
    return rgs - rg_ref
