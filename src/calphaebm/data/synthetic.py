# src/calphaebm/data/synthetic.py

"""Synthetic data generators for testing and debugging."""

import torch
from calphaebm.utils.constants import NUM_AA


def make_extended_chain(
    batch: int,
    length: int,
    bond: float = 3.8,
    noise: float = 0.2,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate an extended Cα chain along x-axis with noise.
    
    Args:
        batch: Batch size.
        length: Chain length.
        bond: Average bond length (Å).
        noise: Gaussian noise amplitude.
        device: Torch device.
        
    Returns:
        (batch, length, 3) coordinates.
    """
    if device is None:
        device = torch.device("cpu")
    
    x = torch.arange(length, device=device, dtype=torch.float32) * bond
    R = torch.zeros((batch, length, 3), device=device, dtype=torch.float32)
    R[..., 0] = x.unsqueeze(0).repeat(batch, 1)
    
    if noise > 0:
        R = R + noise * torch.randn_like(R)
    
    return R


def make_helix(
    batch: int,
    length: int,
    radius: float = 2.3,
    rise: float = 1.5,
    twist: float = 100.0,
    noise: float = 0.1,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate an ideal alpha helix.
    
    Args:
        batch: Batch size.
        length: Chain length.
        radius: Helix radius (Å).
        rise: Rise per residue (Å).
        twist: Twist per residue (degrees).
        noise: Gaussian noise amplitude.
        device: Torch device.
        
    Returns:
        (batch, length, 3) coordinates.
    """
    if device is None:
        device = torch.device("cpu")
    
    twist_rad = torch.deg2rad(torch.tensor(twist, device=device))
    
    i = torch.arange(length, device=device, dtype=torch.float32)
    x = radius * torch.cos(i * twist_rad)
    y = radius * torch.sin(i * twist_rad)
    z = i * rise
    
    R = torch.stack([x, y, z], dim=-1)  # (L, 3)
    R = R.unsqueeze(0).repeat(batch, 1, 1)  # (B, L, 3)
    
    if noise > 0:
        R = R + noise * torch.randn_like(R)
    
    return R


def random_sequence(
    batch: int,
    length: int,
    num_aa: int = NUM_AA,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate random amino acid sequences.
    
    Args:
        batch: Batch size.
        length: Sequence length.
        num_aa: Number of amino acid types.
        device: Torch device.
        
    Returns:
        (batch, length) integer tensor of amino acid indices.
    """
    if device is None:
        device = torch.device("cpu")
    
    return torch.randint(
        low=0,
        high=num_aa,
        size=(batch, length),
        device=device,
        dtype=torch.long,
    )


def random_protein_like(
    batch: int,
    length: int,
    device: torch.device = None,
) -> tuple:
    """Generate random protein-like coordinates and sequence."""
    R = make_extended_chain(batch, length, bond=3.8, noise=0.2, device=device)
    seq = random_sequence(batch, length, device=device)
    return R, seq