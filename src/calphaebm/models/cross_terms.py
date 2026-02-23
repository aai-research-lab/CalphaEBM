# src/calphaebm/models/cross_terms.py

"""Secondary structure cross terms: θ-φ and φ-φ correlations."""

import torch
import torch.nn as nn

from calphaebm.geometry.features import phi_sincos
from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.mlp import MLP


def _cat(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=-1)


class SecondaryStructureEnergy(nn.Module):
    """Secondary structure energy with cross terms.

    E_ss = Σ f_θφ(θ_i, φ_i, context) + Σ f_φφ(φ_i, φ_{i+1}, context)

    Args:
        num_aa: Number of amino acid types.
        emb_dim: Embedding dimension.
        hidden: Hidden layer dimensions.
        use_cos_theta: Use cos(θ) for better numerics.
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

        # θ-φ term: input = [θ_i (or cosθ), sin φ_i, cos φ_i, context]
        self.f_theta_phi = MLP(
            in_dim=1 + 2 + 4 * emb_dim, hidden_dims=hidden, out_dim=1
        )

        # φ-φ term: input = [sin φ_i, cos φ_i, sin φ_{i+1}, cos φ_{i+1}, context]
        self.f_phi_phi = MLP(in_dim=2 + 2 + 4 * emb_dim, hidden_dims=hidden, out_dim=1)

    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute secondary structure energy.

        Args:
            R: (B, L, 3) coordinates.
            seq: (B, L) amino acid indices.

        Returns:
            (B,) Energy per batch element.
        """
        B, L, _ = R.shape
        e = self.emb(seq)

        theta = bond_angles(R)  # (B, L-2)
        phi = torsions(R)  # (B, L-3)
        sc_phi = phi_sincos(phi)  # (B, L-3, 2)

        # θ-φ term (using first L-3 angles)
        theta_i = theta[:, : L - 3]
        theta_in = torch.cos(theta_i) if self.use_cos_theta else theta_i
        ctx = _cat(
            e[:, :-3, :], e[:, 1:-2, :], e[:, 2:-1, :], e[:, 3:, :]
        )  # (B, L-3, 4d)

        x1 = _cat(theta_in.unsqueeze(-1), sc_phi, ctx)
        E1 = self.f_theta_phi(x1).squeeze(-1).sum(dim=1)

        # φ-φ term
        sc_i = sc_phi[:, :-1, :]
        sc_ip1 = sc_phi[:, 1:, :]
        ctx2 = ctx[:, :-1, :]
        x2 = _cat(sc_i, sc_ip1, ctx2)
        E2 = self.f_phi_phi(x2).squeeze(-1).sum(dim=1)

        return E1 + E2

    def energy_from_thetaphi(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute E_ss from precomputed θ and φ.

        Args:
            theta: (B, L-2) bond angles.
            phi: (B, L-3) torsion angles.
            seq: (B, L) amino acid indices.

        Returns:
            (B,) Energy per batch element.
        """
        B, L = seq.shape
        e = self.emb(seq)
        sc_phi = phi_sincos(phi)

        theta_i = theta[:, : L - 3]
        theta_in = torch.cos(theta_i) if self.use_cos_theta else theta_i
        ctx = _cat(e[:, :-3, :], e[:, 1:-2, :], e[:, 2:-1, :], e[:, 3:, :])

        x1 = _cat(theta_in.unsqueeze(-1), sc_phi, ctx)
        E1 = self.f_theta_phi(x1).squeeze(-1).sum(dim=1)

        sc_i = sc_phi[:, :-1, :]
        sc_ip1 = sc_phi[:, 1:, :]
        ctx2 = ctx[:, :-1, :]
        x2 = _cat(sc_i, sc_ip1, ctx2)
        E2 = self.f_phi_phi(x2).squeeze(-1).sum(dim=1)

        return E1 + E2
