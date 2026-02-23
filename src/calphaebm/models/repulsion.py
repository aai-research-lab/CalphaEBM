# src/calphaebm/models/repulsion.py

"""Repulsion energy term for excluded volume."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.models.embeddings import AAEmbedding
from calphaebm.utils.neighbors import topk_nonbonded_pairs
from calphaebm.utils.smooth import smooth_switch


class RepulsionEnergyFixed(nn.Module):
    """Fixed excluded-volume repulsion (analytic wall).

    Energy: E = wall_scale * softplus((r0 - r) / delta) * switch(r)

    Args:
        K: Number of nearest neighbors.
        exclude: Sequence separation cutoff.
        r_on: Switching onset distance (Å).
        r_cut: Cutoff distance (Å).
        r0: Repulsion radius (Å).
        delta: Softness parameter.
        wall_scale: Repulsion strength.
    """

    def __init__(
        self,
        K: int = 64,
        exclude: int = 2,
        r_on: float = 8.0,
        r_cut: float = 10.0,
        r0: float = 4.0,
        delta: float = 0.2,
        wall_scale: float = 10.0,
    ):
        super().__init__()
        self.K = K
        self.exclude = exclude
        self.r_on = float(r_on)
        self.r_cut = float(r_cut)
        self.r0 = float(r0)
        self.delta = float(delta)
        self.wall_scale = float(wall_scale)

    def forward(self, R: torch.Tensor, seq: torch.Tensor | None = None) -> torch.Tensor:
        """Compute repulsion energy.

        Args:
            R: (B, L, 3) coordinates.
            seq: Ignored, for API compatibility.

        Returns:
            (B,) Energy per batch element.
        """
        r, _ = topk_nonbonded_pairs(R, K=self.K, exclude=self.exclude)
        sw = smooth_switch(r, self.r_on, self.r_cut)
        rep = self.wall_scale * F.softplus((self.r0 - r) / (self.delta + 1e-12))
        return (rep * sw).sum(dim=(1, 2))


class RepulsionEnergyLearnedRadius(nn.Module):
    """Learned residue-dependent repulsion radius.

    Each residue type learns an effective radius ρ(s) in [ρ_min, ρ_max].
    Pair radius is r0_ij = ρ(s_i) + ρ(s_j).

    Energy: E = wall_scale * softplus((r0_ij - r_ij) / delta) * switch(r_ij)

    Args:
        num_aa: Number of amino acid types.
        emb_dim: Embedding dimension.
        K: Number of nearest neighbors.
        exclude: Sequence separation cutoff.
        r_on: Switching onset distance.
        r_cut: Cutoff distance.
        delta: Softness parameter.
        wall_scale: Repulsion strength.
        rho_min: Minimum effective radius.
        rho_max: Maximum effective radius.
        rho_base: Initial radius for all residues.
    """

    def __init__(
        self,
        num_aa: int = 20,
        emb_dim: int = 16,
        K: int = 64,
        exclude: int = 2,
        r_on: float = 8.0,
        r_cut: float = 10.0,
        delta: float = 0.3,
        wall_scale: float = 10.0,
        rho_min: float = 1.6,
        rho_max: float = 2.8,
        rho_base: float = 2.0,
    ):
        super().__init__()
        self.K = K
        self.exclude = exclude
        self.r_on = float(r_on)
        self.r_cut = float(r_cut)
        self.delta = float(delta)
        self.wall_scale = float(wall_scale)

        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.rho_base = float(rho_base)

        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)
        self.size_head = nn.Linear(emb_dim, 1)

        # Initialize to rho_base
        with torch.no_grad():
            self.size_head.weight.zero_()
            frac = (self.rho_base - self.rho_min) / max(
                1e-6, (self.rho_max - self.rho_min)
            )
            frac = float(min(0.999, max(0.001, frac)))
            bias = math.log(frac / (1.0 - frac))
            self.size_head.bias.fill_(bias)

    def rho(self, seq: torch.Tensor) -> torch.Tensor:
        """Compute per-residue radii.

        Args:
            seq: (B, L) amino acid indices.

        Returns:
            (B, L) radii in Å.
        """
        e = self.emb(seq)
        x = self.size_head(e).squeeze(-1)
        frac = torch.sigmoid(x)
        return self.rho_min + (self.rho_max - self.rho_min) * frac

    def pair_r0(self, seq_i: torch.Tensor, seq_j: torch.Tensor) -> torch.Tensor:
        """Compute pair radii.

        Args:
            seq_i: (B, L, K) indices for i.
            seq_j: (B, L, K) indices for j.

        Returns:
            (B, L, K) pair radii.
        """
        rho_i = self.rho(seq_i)
        rho_j = self.rho(seq_j)
        return rho_i + rho_j

    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute repulsion energy.

        Args:
            R: (B, L, 3) coordinates.
            seq: (B, L) amino acid indices.

        Returns:
            (B,) Energy per batch element.
        """
        r, j = topk_nonbonded_pairs(R, K=self.K, exclude=self.exclude)
        B, L, K = j.shape

        # Get sequences for i and j
        seq_i = seq.unsqueeze(-1).expand(-1, -1, K)
        seq_j = torch.gather(seq.unsqueeze(-1).expand(-1, -1, K), 1, j)

        r0 = self.pair_r0(seq_i, seq_j)
        sw = smooth_switch(r, self.r_on, self.r_cut)
        rep = self.wall_scale * F.softplus((r0 - r) / (self.delta + 1e-12))

        return (rep * sw).sum(dim=(1, 2))


# Alias for default
RepulsionEnergy = RepulsionEnergyLearnedRadius
