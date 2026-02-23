# src/calphaebm/models/local_terms.py

"""Local energy term: bond lengths, angles, torsions."""

import torch
import torch.nn as nn

from calphaebm.geometry.internal import bond_lengths, bond_angles, torsions
from calphaebm.geometry.features import phi_sincos
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.mlp import MLP


def _cat(*xs: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors along last dimension."""
    return torch.cat(xs, dim=-1)


class LocalEnergy(nn.Module):
    """Local backbone energy: E_local = Σ f_l(ℓ) + Σ f_θ(θ) + Σ f_φ(φ).
    
    Args:
        num_aa: Number of amino acid types.
        emb_dim: Embedding dimension for sequence context.
        hidden: Hidden layer dimensions for MLPs.
        use_cos_theta: If True, use cos(θ) instead of θ for better numerics.
    """
    
    def __init__(
        self,
        num_aa: int = 20,
        emb_dim: int = 16,
        hidden: tuple = (128, 128),
        use_cos_theta: bool = True,
    ):
        super().__init__()
        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)
        self.use_cos_theta = use_cos_theta
        
        # Bond length MLP: input = [ℓ_i, e_i, e_{i+1}]
        self.f_l = MLP(in_dim=1 + 2 * emb_dim, hidden_dims=hidden, out_dim=1)
        
        # Bond angle MLP: input = [θ_i, e_{i-1}, e_i, e_{i+1}]
        self.f_theta = MLP(in_dim=1 + 3 * emb_dim, hidden_dims=hidden, out_dim=1)
        
        # Torsion MLP: input = [sin φ_i, cos φ_i, e_{i-1}, e_i, e_{i+1}, e_{i+2}]
        self.f_phi = MLP(in_dim=2 + 4 * emb_dim, hidden_dims=hidden, out_dim=1)
        
    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute local energy.
        
        Args:
            R: (B, L, 3) Cartesian coordinates.
            seq: (B, L) Amino acid indices.
            
        Returns:
            (B,) Energy per batch element.
        """
        B, L, _ = R.shape
        e = self.emb(seq)  # (B, L, d)
        
        # Bond lengths ℓ_i (i=0..L-2)
        l = bond_lengths(R)  # (B, L-1)
        e_l = _cat(e[:, :-1, :], e[:, 1:, :])  # (B, L-1, 2d)
        x_l = _cat(l.unsqueeze(-1), e_l)
        E_l = self.f_l(x_l).squeeze(-1).sum(dim=1)
        
        # Bond angles θ_i (i=1..L-2)
        theta = bond_angles(R)  # (B, L-2)
        theta_in = torch.cos(theta) if self.use_cos_theta else theta
        e_theta = _cat(e[:, :-2, :], e[:, 1:-1, :], e[:, 2:, :])  # (B, L-2, 3d)
        x_theta = _cat(theta_in.unsqueeze(-1), e_theta)
        E_theta = self.f_theta(x_theta).squeeze(-1).sum(dim=1)
        
        # Torsions φ_i (i=1..L-3)
        phi = torsions(R)  # (B, L-3)
        sc = phi_sincos(phi)  # (B, L-3, 2)
        e_phi = _cat(e[:, :-3, :], e[:, 1:-2, :], e[:, 2:-1, :], e[:, 3:, :])  # (B, L-3, 4d)
        x_phi = _cat(sc, e_phi)
        E_phi = self.f_phi(x_phi).squeeze(-1).sum(dim=1)
        
        return E_l + E_theta + E_phi
    
    def energy_from_internals(
        self,
        l: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local energy from precomputed internal coordinates.
        
        Args:
            l: (B, L-1) Bond lengths.
            theta: (B, L-2) Bond angles.
            phi: (B, L-3) Torsions.
            seq: (B, L) Amino acid indices.
            
        Returns:
            (B,) Energy per batch element.
        """
        B, L = seq.shape
        e = self.emb(seq)
        
        # Bond lengths
        e_l = _cat(e[:, :-1, :], e[:, 1:, :])
        x_l = _cat(l.unsqueeze(-1), e_l)
        E_l = self.f_l(x_l).squeeze(-1).sum(dim=1)
        
        # Bond angles
        theta_in = torch.cos(theta) if self.use_cos_theta else theta
        e_theta = _cat(e[:, :-2, :], e[:, 1:-1, :], e[:, 2:, :])
        x_theta = _cat(theta_in.unsqueeze(-1), e_theta)
        E_theta = self.f_theta(x_theta).squeeze(-1).sum(dim=1)
        
        # Torsions
        sc = phi_sincos(phi)
        e_phi = _cat(e[:, :-3, :], e[:, 1:-2, :], e[:, 2:-1, :], e[:, 3:, :])
        x_phi = _cat(sc, e_phi)
        E_phi = self.f_phi(x_phi).squeeze(-1).sum(dim=1)
        
        return E_l + E_theta + E_phi