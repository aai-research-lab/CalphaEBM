# src/calphaebm/models/embeddings.py

"""Amino acid embeddings for sequence conditioning."""

import torch
import torch.nn as nn


class AAEmbedding(nn.Module):
    """Learnable embedding for amino acid types.

    Maps integer amino acid indices (0-19) to dense vectors.

    Args:
        num_aa: Number of amino acid types (default 20).
        dim: Embedding dimension.
    """

    def __init__(self, num_aa: int = 20, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_aa, dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Convert sequence to embeddings.

        Args:
            seq: (B, L) integer tensor of amino acid indices.

        Returns:
            (B, L, dim) embedding tensor.
        """
        return self.emb(seq)
