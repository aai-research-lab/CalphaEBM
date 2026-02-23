"""Mathematical utilities with safe numerical operations."""

from typing import Tuple, Union

import numpy as np
import torch


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute norm with numerical safety (avoid zero gradient).

    Args:
        x: Input tensor.
        dim: Dimension along which to compute norm.
        keepdim: Keep reduced dimension.
        eps: Small value to avoid sqrt(0).

    Returns:
        Norm of x along specified dimension.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)

def wrap_to_pi(phi: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi] range using atan2(sin, cos).

    This is the most numerically stable method.

    Args:
        phi: Angle in radians.

    Returns:
        Angle wrapped to [-pi, pi].
    """
    return torch.atan2(torch.sin(phi), torch.cos(phi))

def wrap_to_2pi(phi: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [0, 2pi) range.

    Args:
        phi: Angle in radians.

    Returns:
        Angle wrapped to [0, 2pi).
    """
    return phi % (2 * torch.pi)


def kabsch_rotate(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal rotation aligning P onto Q.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        (P_aligned, rotation_matrix) where P_aligned = (P - P_com) @ R
    """
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    C = P_centered.T @ Q_centered
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt

    P_aligned = P_centered @ R
    return P_aligned, R


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between two point sets after optimal alignment.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        RMSD value in same units as input.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    P_aligned, _ = kabsch_rotate(P, Q)
    Q_centered = Q - Q.mean(axis=0)
    diff = P_aligned - Q_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def center_of_mass(R: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Compute center of mass (equal weights).

    Args:
        R: (..., N, 3) coordinates.

    Returns:
        Center of mass with same type as input.
    """
    return R.mean(axis=-2)
