# src/calphaebm/training/losses/contrastive.py

"""Contrastive losses for energy-based training."""

import torch


def contrastive_logistic_loss(
    E_pos: torch.Tensor,
    E_neg: torch.Tensor,
) -> torch.Tensor:
    """Logistic (binary cross-entropy) loss for contrastive learning.

    Encourages E_pos < E_neg by optimizing:
        loss = log(1 + exp(E_pos - E_neg))

    Args:
        E_pos: (B,) Energies of positive samples (should be low).
        E_neg: (B,) Energies of negative samples (should be high).

    Returns:
        Scalar loss.
    """
    return torch.log1p(torch.exp(E_pos - E_neg)).mean()


def contrastive_hinge_loss(
    E_pos: torch.Tensor,
    E_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge loss for contrastive learning.

    loss = max(0, E_pos - E_neg + margin)

    Args:
        E_pos: (B,) Energies of positive samples.
        E_neg: (B,) Energies of negative samples.
        margin: Minimum energy gap.

    Returns:
        Scalar loss.
    """
    return torch.clamp(E_pos - E_neg + margin, min=0.0).mean()


def contrastive_ranking_loss(
    E_pos: torch.Tensor,
    E_negs: torch.Tensor,  # (B, N_neg)
    margin: float = 1.0,
) -> torch.Tensor:
    """Ranking loss with multiple negatives.

    loss = mean(max(0, E_pos - E_neg + margin) over all negatives)

    Args:
        E_pos: (B,) Positive energies.
        E_negs: (B, N_neg) Negative energies.
        margin: Minimum energy gap.

    Returns:
        Scalar loss.
    """
    diff = E_pos.unsqueeze(-1) - E_negs + margin
    return torch.clamp(diff, min=0.0).mean()
