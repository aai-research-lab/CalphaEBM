# src/calphaebm/utils/neighbors.py

"""Neighbor list utilities for efficient nonbonded calculations."""

import torch
import numpy as np
from typing import Tuple, Optional


def pairwise_distances(R: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise distances.
    
    Args:
        R: (B, L, 3) coordinates.
        
    Returns:
        (B, L, L) distance matrix.
    """
    return torch.cdist(R, R)


def topk_nonbonded_pairs(
    R: torch.Tensor,
    K: int = 64,
    exclude: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find K nearest nonbonded neighbors for each residue.
    
    Args:
        R: (B, L, 3) coordinates.
        K: Number of neighbors to return.
        exclude: Sequence separation cutoff (|i-j| <= exclude excluded).
        
    Returns:
        (distances, indices):
            distances: (B, L, K) distances to K nearest nonbonded neighbors.
            indices: (B, L, K) indices of those neighbors.
    """
    B, L, _ = R.shape
    D = pairwise_distances(R)  # (B, L, L)
    
    # Create sequence separation mask
    idx = torch.arange(L, device=R.device)
    ii = idx.view(1, L, 1)
    jj = idx.view(1, 1, L)
    mask = (torch.abs(ii - jj) <= exclude)
    
    # Mask out excluded pairs
    D_masked = D.masked_fill(mask, float("inf"))
    
    # Get K smallest (excluding inf)
    K_actual = min(K, L - exclude - 1)
    distances, indices = torch.topk(D_masked, k=K_actual, dim=-1, largest=False)
    
    # If K_actual < K, pad with zeros
    if K_actual < K:
        pad_d = torch.full((B, L, K - K_actual), float("inf"), device=R.device)
        pad_i = torch.zeros((B, L, K - K_actual), dtype=torch.long, device=R.device)
        distances = torch.cat([distances, pad_d], dim=-1)
        indices = torch.cat([indices, pad_i], dim=-1)
    
    return distances, indices


class NeighborList:
    """Verlet-style neighbor list with skin."""
    
    def __init__(
        self,
        cutoff: float = 12.0,
        skin: float = 2.0,
        max_neighbors: int = 64,
    ):
        self.cutoff = cutoff
        self.skin = skin
        self.max_neighbors = max_neighbors
        self.pairs = None
        self.last_positions = None
        
    def update(
        self,
        R: torch.Tensor,
        exclude: int = 2,
        force: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update neighbor list if needed.
        
        Args:
            R: (B, L, 3) current positions.
            exclude: Sequence separation cutoff.
            force: Force update even if skin not exceeded.
            
        Returns:
            (distances, indices) as in topk_nonbonded_pairs.
        """
        needs_update = force
        if not needs_update and self.last_positions is not None:
            drift = torch.norm(R - self.last_positions, dim=-1).max().item()
            if drift > self.skin:
                needs_update = True
                
        if needs_update:
            self.pairs = topk_nonbonded_pairs(
                R, K=self.max_neighbors, exclude=exclude
            )
            self.last_positions = R.clone()
            
        return self.pairs