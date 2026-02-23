# src/calphaebm/geometry/dihedral.py

"""Stable dihedral (torsion) angle implementation.

Returns dihedral in (-pi, pi]. Designed to be differentiable.
Reference: Common MD implementations using atan2 with plane normals.
"""

import torch
from calphaebm.utils.math import safe_norm, wrap_to_pi


def dihedral(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angle for points p0-p1-p2-p3.
    
    The dihedral angle is the angle between the planes (p0,p1,p2) and (p1,p2,p3).
    
    Args:
        p0, p1, p2, p3: Points with shape (..., 3).
        
    Returns:
        Dihedral angles in radians, shape (...,), range [-pi, pi].
    """
    # Vectors along bonds
    b0 = p1 - p0  # (p0->p1)
    b1 = p2 - p1  # (p1->p2)
    b2 = p3 - p2  # (p2->p3)
    
    # Normalize b1 for projection
    b1_norm = safe_norm(b1, dim=-1, keepdim=True)
    b1_unit = b1 / (b1_norm + 1e-12)
    
    # Components perpendicular to b1
    v = b0 - (torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit)
    w = b2 - (torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit)
    
    # Compute angle using atan2 for numerical stability
    x = torch.sum(v * w, dim=-1)  # cos component
    y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)  # sin component
    
    phi = torch.atan2(y, x)
    return wrap_to_pi(phi)


def dihedral_from_points(points: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angles for all quadruplets in a chain.
    
    Args:
        points: (..., L, 3) coordinates.
        
    Returns:
        (..., L-3) dihedral angles.
    """
    p0 = points[..., :-3, :]
    p1 = points[..., 1:-2, :]
    p2 = points[..., 2:-1, :]
    p3 = points[..., 3:, :]
    return dihedral(p0, p1, p2, p3)