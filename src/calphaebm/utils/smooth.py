# src/calphaebm/utils/smooth.py

"""Smooth switching functions for nonbonded interactions."""

import torch


def smooth_switch(
    r: torch.Tensor,
    r_on: float,
    r_cut: float,
) -> torch.Tensor:
    """Cosine switching function from r_on to r_cut.
    
    s(r) = 1 for r <= r_on
    s(r) = 0.5*(1 + cos(pi*(r - r_on)/(r_cut - r_on))) for r_on < r < r_cut
    s(r) = 0 for r >= r_cut
    
    Args:
        r: Distances.
        r_on: Onset distance for switching.
        r_cut: Cutoff distance.
        
    Returns:
        Switching values in [0, 1].
    """
    s = torch.ones_like(r)
    
    # Zero beyond cutoff
    s = torch.where(r >= r_cut, torch.zeros_like(r), s)
    
    # Smooth region
    mid = (r > r_on) & (r < r_cut)
    x = (r - r_on) / (r_cut - r_on + 1e-12)
    s_mid = 0.5 * (1.0 + torch.cos(torch.pi * x))
    s = torch.where(mid, s_mid, s)
    
    return s


def smooth_switch_derivative(
    r: torch.Tensor,
    r_on: float,
    r_cut: float,
) -> torch.Tensor:
    """Derivative of smooth_switch with respect to r."""
    ds = torch.zeros_like(r)
    
    mid = (r > r_on) & (r < r_cut)
    if mid.any():
        x = (r - r_on) / (r_cut - r_on + 1e-12)
        ds_mid = -0.5 * torch.pi * torch.sin(torch.pi * x) / (r_cut - r_on + 1e-12)
        ds = torch.where(mid, ds_mid, ds)
        
    return ds


def smooth_attractive(
    r: torch.Tensor,
    r0: float,
    width: float = 0.5,
    depth: float = 1.0,
) -> torch.Tensor:
    """Smooth attractive well (Lennard-Jones-like but differentiable).
    
    V(r) = -depth * exp(-(r - r0)^2 / (2 * width^2))
    
    Args:
        r: Distances.
        r0: Well minimum position.
        width: Well width.
        depth: Well depth (positive).
        
    Returns:
        Attractive energy (negative).
    """
    return -depth * torch.exp(-0.5 * ((r - r0) / width) ** 2)