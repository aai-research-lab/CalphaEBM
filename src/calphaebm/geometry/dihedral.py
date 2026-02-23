"""Stable dihedral (torsion) angle implementation."""

import torch

from calphaebm.utils.math import safe_norm, wrap_to_pi


def dihedral(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angle for points p0-p1-p2-p3."""
    # Ensure all inputs are float32
    p0 = p0.float()
    p1 = p1.float()
    p2 = p2.float()
    p3 = p3.float()

    # Vectors along bonds
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Standard MD formula
    u = torch.cross(b0, b1, dim=-1)
    v = torch.cross(b1, b2, dim=-1)
    w = torch.cross(u, b1, dim=-1)

    # Normalize for stability
    u_norm = safe_norm(u, dim=-1, keepdim=True)
    v_norm = safe_norm(v, dim=-1, keepdim=True)
    w_norm = safe_norm(w, dim=-1, keepdim=True)

    u = torch.where(u_norm > 1e-8, u / u_norm, torch.zeros_like(u))
    v = torch.where(v_norm > 1e-8, v / v_norm, torch.zeros_like(v))
    w = torch.where(w_norm > 1e-8, w / w_norm, torch.zeros_like(w))

    x = torch.sum(u * v, dim=-1)
    y = torch.sum(w * v, dim=-1)

    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)

    phi = torch.atan2(y, x)
    return wrap_to_pi(phi)


def dihedral_from_points(points: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angles for all quadruplets in a chain."""
    points = points.float()
    p0 = points[..., :-3, :]
    p1 = points[..., 1:-2, :]
    p2 = points[..., 2:-1, :]
    p3 = points[..., 3:, :]
    return dihedral(p0, p1, p2, p3)
