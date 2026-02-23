"""Local neural energy terms for a Cα backbone.

E_local(R|s) = Σ f_l(ℓ_i, context) + Σ f_θ(θ_i, context) + Σ f_φ(φ_i, context)
All computed from Cartesian coordinates so constraints are always satisfied.
"""

import torch
import torch.nn as nn

from calphaebm.geometry.features import phi_sincos
from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.mlp import MLP


def _cat(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=-1)


class LocalEnergy(nn.Module):
    def __init__(
        self,
        num_aa: int = 20,
        emb_dim: int = 16,
        hidden=(128, 128),
        use_cos_theta: bool = True,
    ):
        super().__init__()
        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)
        self.use_cos_theta = use_cos_theta

        self.f_l = MLP(in_dim=1 + 2 * emb_dim, hidden_dims=hidden, out_dim=1)
        self.f_theta = MLP(in_dim=1 + 3 * emb_dim, hidden_dims=hidden, out_dim=1)
        self.f_phi = MLP(in_dim=2 + 4 * emb_dim, hidden_dims=hidden, out_dim=1)

    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """R: [B,L,3], seq: [B,L] -> energy [B]"""
        B, L, _ = R.shape
        e = self.emb(seq)  # [B,L,d]

        # Bond lengths ℓ_i (i=0..L-2)
        lengths = bond_lengths(R)  # [B,L-1]
        e_l = _cat(e[:, :-1, :], e[:, 1:, :])  # [B,L-1,2d]
        x_l = _cat(lengths.unsqueeze(-1), e_l)
        E_l = self.f_l(x_l).squeeze(-1).sum(dim=1)

        # Bond angles θ_i (i=1..L-2)
        theta = bond_angles(R)  # [B,L-2]
        theta_in = torch.cos(theta) if self.use_cos_theta else theta
        e_theta = _cat(e[:, :-2, :], e[:, 1:-1, :], e[:, 2:, :])  # [B,L-2,3d]
        x_theta = _cat(theta_in.unsqueeze(-1), e_theta)
        E_theta = self.f_theta(x_theta).squeeze(-1).sum(dim=1)

        # Torsions φ_i (i=1..L-3)
        phi = torsions(R)  # [B,L-3]
        sc = phi_sincos(phi)  # [B,L-3,2]
        e_phi = _cat(
            e[:, :-3, :], e[:, 1:-2, :], e[:, 2:-1, :], e[:, 3:, :]
        )  # [B,L-3,4d]
        x_phi = _cat(sc, e_phi)
        E_phi = self.f_phi(x_phi).squeeze(-1).sum(dim=1)

        return E_l + E_theta + E_phi
