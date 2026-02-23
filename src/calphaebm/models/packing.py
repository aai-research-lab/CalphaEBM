# src/calphaebm/models/packing.py

"""Packing energy term for long-range organization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from calphaebm.geometry.features import distance_features
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.mlp import MLP
from calphaebm.utils.neighbors import topk_nonbonded_pairs
from calphaebm.utils.smooth import smooth_switch


class PackingEnergy(nn.Module):
    """Packing energy with residue-pair specific attraction.

    E_pack = -Σ w_ij * φ(r_ij) * switch(r_ij)
    where φ are RBF features and w_ij are learned pair-specific weights.

    Args:
        num_aa: Number of amino acid types.
        emb_dim: Embedding dimension.
        hidden_pair: Hidden dimensions for pair MLP.
        K: Number of nearest neighbors.
        exclude: Sequence separation cutoff.
        r_on: Switching onset distance.
        r_cut: Cutoff distance.
        rbf_centers: Centers for RBF features.
        rbf_width: Width for RBF features.
    """

    def __init__(
        self,
        num_aa: int = 20,
        emb_dim: int = 16,
        hidden_pair: tuple = (128, 128),
        K: int = 64,
        exclude: int = 3,
        r_on: float = 10.0,
        r_cut: float = 12.0,
        rbf_centers: tuple = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0),
        rbf_width: float = 0.5,
    ):
        super().__init__()
        self.K = K
        self.exclude = exclude
        self.r_on = float(r_on)
        self.r_cut = float(r_cut)

        self.emb = AAEmbedding(num_aa=num_aa, dim=emb_dim)

        # RBF centers
        M = len(rbf_centers)
        self.register_buffer("centers", torch.tensor(rbf_centers, dtype=torch.float32))
        self.register_buffer(
            "widths", torch.full((M,), float(rbf_width), dtype=torch.float32)
        )

        # Pair MLP: input = [e_i, e_j, e_i * e_j] -> M weights
        self.pair_mlp = MLP(in_dim=3 * emb_dim, hidden_dims=hidden_pair, out_dim=M)

    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute packing energy.

        Args:
            R: (B, L, 3) coordinates.
            seq: (B, L) amino acid indices.

        Returns:
            (B,) Energy per batch element.
        """
        B, L, _ = R.shape
        e = self.emb(seq)  # (B, L, d)

        # Get neighbor pairs
        r, j = topk_nonbonded_pairs(R, K=self.K, exclude=self.exclude)  # (B, L, K)

        # Gather embeddings for neighbors
        d = e.shape[-1]
        j_exp = j.unsqueeze(-1).expand(-1, -1, -1, d)
        e_j = torch.gather(e.unsqueeze(2).expand(-1, -1, j.shape[2], -1), 1, j_exp)
        e_i = e.unsqueeze(2).expand_as(e_j)

        # Pair features
        pair_feat = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
        w = F.softplus(self.pair_mlp(pair_feat))  # (B, L, K, M)

        # RBF features
        phi = distance_features(r, self.centers.to(r.device), self.widths.to(r.device))

        # Attractive energy (negative)
        att = -(w * phi).sum(dim=-1)

        # Smooth switching
        sw = smooth_switch(r, self.r_on, self.r_cut)

        return (att * sw).sum(dim=(1, 2))
