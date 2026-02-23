"""Differentiable internal coordinates from Cartesian Cα coordinates.

R is expected to be [B, L, 3] or [L, 3].
Outputs are batched if input is batched.
"""

import numpy as np
import torch

from calphaebm.geometry.dihedral import dihedral
from calphaebm.utils.math import safe_norm


def _ensure_batch(R):
    """Convert 2D input to batched format."""
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if R.dim() == 2:
        return R.unsqueeze(0)
    return R


def bond_lengths(R: torch.Tensor) -> torch.Tensor:
    """Compute bond lengths ℓ_i = ||r_{i+1} - r_i||.

    Args:
        R: (B, L, 3) or (L, 3) coordinates.

    Returns:
        (B, L-1) bond lengths in Å.
    """
    Rb = _ensure_batch(R)
    diffs = Rb[:, 1:, :] - Rb[:, :-1, :]
    return safe_norm(diffs, dim=-1)


def bond_angles(R: torch.Tensor) -> torch.Tensor:
    """Compute bond angles θ_i at i=1..L-2 using (i-1,i,i+1).

    Args:
        R: (B, L, 3) or (L, 3) coordinates.

    Returns:
        (B, L-2) bond angles in radians.
    """
    Rb = _ensure_batch(R)

    # Vectors from central atom to neighbors
    u = Rb[:, :-2, :] - Rb[:, 1:-1, :]  # r_{i-1} - r_i
    v = Rb[:, 2:, :] - Rb[:, 1:-1, :]   # r_{i+1} - r_i

    # Compute cosine of angle
    u_norm = safe_norm(u, dim=-1)
    v_norm = safe_norm(v, dim=-1)
    cos_theta = torch.sum(u * v, dim=-1) / (u_norm * v_norm + 1e-12)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    return torch.acos(cos_theta)


def torsions(R: torch.Tensor) -> torch.Tensor:
    """Compute torsion angles φ_i for quadruplets (i-1,i,i+1,i+2).

    Args:
        R: (B, L, 3) or (L, 3) coordinates.

    Returns:
        (B, L-3) torsion angles in radians.
    """
    Rb = _ensure_batch(R)

    p0 = Rb[:, :-3, :]
    p1 = Rb[:, 1:-2, :]
    p2 = Rb[:, 2:-1, :]
    p3 = Rb[:, 3:, :]

    return dihedral(p0, p1, p2, p3)


def all_internal(R: torch.Tensor) -> dict:
    """Compute all internal coordinates at once.

    Args:
        R: (B, L, 3) or (L, 3) coordinates.

    Returns:
        Dictionary with keys:
            - 'l': bond lengths
            - 'theta': bond angles
            - 'phi': torsion angles
    """
    Rb = _ensure_batch(R)
    return {
        "l": bond_lengths(Rb),
        "theta": bond_angles(Rb),
        "phi": torsions(Rb),
    }


def check_geometry(R, max_jump: float = 4.5) -> dict:
    """Check geometric sanity of a structure.

    Args:
        R: (L, 3) coordinates (numpy or torch tensor).
        max_jump: Maximum allowed Cα-Cα distance.

    Returns:
        Dictionary with validation results.
    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    R = _ensure_batch(R)[0]
    L = R.shape[0]

    # Check bond lengths
    l = bond_lengths(R)
    l_min = l.min().item()
    l_max = l.max().item()
    l_mean = l.mean().item()

    # Check for chain breaks
    broken = (l > max_jump).sum().item()

    # Check bond angles (should be ~90-140° for proteins)
    theta = bond_angles(R) * 180 / torch.pi
    theta_min = theta.min().item()
    theta_max = theta.max().item()

    # Check for steric clashes (very rough)
    from calphaebm.utils.neighbors import pairwise_distances
    D = pairwise_distances(R.unsqueeze(0))[0]
    # Exclude bonded pairs
    for i in range(L):
        D[i, max(0, i-2):min(L, i+3)] = float("inf")
    min_nonbonded = D.min().item()

    return {
        "length": L,
        "bond_lengths": {"min": l_min, "max": l_max, "mean": l_mean},
        "broken_chain": broken > 0,
        "bond_angles_deg": {"min": theta_min, "max": theta_max},
        "min_nonbonded": min_nonbonded,
        "valid": min_nonbonded > 2.5 and l_max < max_jump,
    }
