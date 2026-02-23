# src/calphaebm/evaluation/metrics/rdf.py

"""Radial distribution function (RDF) calculation."""

from typing import Tuple

import numpy as np


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances."""
    diff = R[:, None, :] - R[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1) + 1e-12)


def rdf_counts(
    R: np.ndarray,
    r_max: float = 20.0,
    dr: float = 0.25,
    exclude: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RDF counts histogram.

    Args:
        R: (N, 3) coordinates.
        r_max: Maximum distance (Å).
        dr: Bin width (Å).
        exclude: Sequence separation cutoff.

    Returns:
        (centers, counts) where centers are bin centers, counts are histogram.
    """
    D = pairwise_distances(R)
    N = D.shape[0]

    # Mask out excluded pairs
    mask = np.ones((N, N), dtype=bool)
    for k in range(-exclude, exclude + 1):
        mask &= ~np.eye(N, k=k, dtype=bool)

    iu = np.triu_indices(N, k=1)
    vals = D[iu][mask[iu]]

    bins = np.arange(0.0, r_max + dr, dr)
    hist, edges = np.histogram(vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, hist.astype(np.float64)


def rdf_normalized(
    centers: np.ndarray,
    counts: np.ndarray,
    dr: float,
    tail_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize RDF counts to get g(r).

    Two-step normalization:
    1. Shell normalization: counts / (4πr² dr)
    2. Tail normalization: scale so tail mean = 1

    Args:
        centers: Bin centers.
        counts: Raw counts per bin.
        dr: Bin width.
        tail_frac: Fraction of tail to use for normalization.

    Returns:
        (g_norm, g_raw, tail_mean) where g_norm has tail mean = 1.
    """
    # Shell normalization
    shell_vol = 4.0 * np.pi * (centers**2) * dr
    g_raw = counts / (shell_vol + 1e-12)

    # Tail normalization
    k0 = int(np.floor((1.0 - tail_frac) * len(g_raw)))
    k0 = max(k0, 1)
    tail = g_raw[k0:]
    tail_mean = float(np.mean(tail)) if np.any(tail > 0) else 1.0

    g_norm = g_raw / (tail_mean + 1e-12)

    return g_norm, g_raw, tail_mean


def batch_rdf(
    trajectory: np.ndarray,
    r_max: float = 20.0,
    dr: float = 0.25,
    exclude: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average RDF over trajectory.

    Args:
        trajectory: (n_frames, N, 3) coordinates.
        r_max, dr, exclude: As in rdf_counts.

    Returns:
        (centers, mean_counts, mean_g_norm)
    """
    n_frames = trajectory.shape[0]
    accum_counts = None
    centers = None

    for i in range(n_frames):
        c, h = rdf_counts(trajectory[i], r_max, dr, exclude)
        if accum_counts is None:
            centers = c
            accum_counts = h.copy()
        else:
            accum_counts += h

    mean_counts = accum_counts / n_frames
    g_norm, g_raw, tail_mean = rdf_normalized(centers, mean_counts, dr)

    return centers, mean_counts, g_norm
