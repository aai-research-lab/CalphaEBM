# src/calphaebm/geometry/features.py

"""Feature transformations for geometric quantities."""

import torch


def phi_sincos(phi: torch.Tensor) -> torch.Tensor:
    """Convert torsion angles to sin/cos features.

    This transformation handles periodicity and avoids discontinuities.

    Args:
        phi: (..., L) torsion angles in radians.

    Returns:
        (..., L, 2) tensor with [sin(phi), cos(phi)].
    """
    return torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)


def theta_features(theta: torch.Tensor, use_cos: bool = True) -> torch.Tensor:
    """Convert bond angles to features.

    Args:
        theta: Bond angles in radians.
        use_cos: If True, return cos(theta) for better numerics.

    Returns:
        Features of shape (..., 1).
    """
    if use_cos:
        return torch.cos(theta).unsqueeze(-1)
    return theta.unsqueeze(-1)


def distance_features(
    r: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
) -> torch.Tensor:
    """Radial basis function features for distances.

    Args:
        r: Distances.
        centers: RBF centers.
        widths: RBF widths.

    Returns:
        RBF features of shape (..., len(centers)).
    """
    diff = r.unsqueeze(-1) - centers.view(*([1] * r.dim()), -1)
    return torch.exp(-0.5 * (diff / (widths.view(*([1] * r.dim()), -1) + 1e-12)) ** 2)
