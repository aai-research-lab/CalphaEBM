# src/calphaebm/training/balancing.py

"""Force-scale balancing to recommend lambda weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from calphaebm.models.energy import TotalEnergy
from calphaebm.utils.math import safe_norm


@dataclass
class BalanceReport:
    """Report from lambda balancing."""

    force_scales: Dict[str, float]  # Median force norm per term
    recommended_lambdas: Dict[str, float]  # Recommended lambda values


def _median_force_scale(F: torch.Tensor) -> float:
    """Compute median force norm across all atoms."""
    norms = safe_norm(F, dim=-1).reshape(-1)
    return float(torch.median(norms).item())


def term_forces(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute forces from each energy term separately.

    Args:
        model: TotalEnergy model.
        R: (B, L, 3) coordinates.
        seq: (B, L) amino acid indices.

    Returns:
        Dictionary mapping term names to force tensors.
    """
    forces: Dict[str, torch.Tensor] = {}

    # Local term
    Rg = R.detach().requires_grad_(True)
    E_local = model.local(Rg, seq).sum()
    F_local = -torch.autograd.grad(E_local, Rg, create_graph=False)[0]
    forces["local"] = F_local.detach()

    # Repulsion
    if model.repulsion is not None:
        Rg = R.detach().requires_grad_(True)
        E_rep = model.repulsion(Rg, seq).sum()
        F_rep = -torch.autograd.grad(E_rep, Rg, create_graph=False)[0]
        forces["repulsion"] = F_rep.detach()

    # Secondary
    if model.secondary is not None:
        Rg = R.detach().requires_grad_(True)
        E_ss = model.secondary(Rg, seq).sum()
        F_ss = -torch.autograd.grad(E_ss, Rg, create_graph=False)[0]
        forces["secondary"] = F_ss.detach()

    # Packing
    if model.packing is not None:
        Rg = R.detach().requires_grad_(True)
        E_pack = model.packing(Rg, seq).sum()
        F_pack = -torch.autograd.grad(E_pack, Rg, create_graph=False)[0]
        forces["packing"] = F_pack.detach()

    return forces


def recommend_lambdas(
    model: TotalEnergy,
    R: torch.Tensor,
    seq: torch.Tensor,
    reference: str = "local",
    current_lambdas: Optional[Dict[str, float]] = None,
    clip: Tuple[float, float] = (1e-3, 1e3),
) -> BalanceReport:
    """Recommend lambda values to balance force scales.

    Computes median force norm for each term, then recommends lambdas
    such that all terms have the same force scale as the reference term.

    Args:
        model: TotalEnergy model.
        R: (B, L, 3) coordinates.
        seq: (B, L) amino acid indices.
        reference: Term to use as reference scale.
        current_lambdas: Current lambda values (default: all 1.0).
        clip: Min/max allowed lambda values.

    Returns:
        BalanceReport with force scales and recommended lambdas.
    """
    forces = term_forces(model, R, seq)
    scales = {k: _median_force_scale(F) for k, F in forces.items()}

    if reference not in scales or scales[reference] <= 0:
        raise ValueError(
            f"Reference term {reference} missing/zero. Have {list(scales)}"
        )

    # Current lambdas (default 1.0)
    lam = current_lambdas or {k: 1.0 for k in scales.keys()}
    ref_target = lam.get(reference, 1.0) * scales[reference]

    # Recommend new lambdas
    rec = {}
    lo, hi = clip
    for k, s in scales.items():
        if s <= 0:
            rec[k] = lam.get(k, 1.0)
        else:
            rec[k] = float(max(lo, min(hi, ref_target / s)))

    return BalanceReport(
        force_scales=scales,
        recommended_lambdas=rec,
    )


def balance_across_batch(
    model: TotalEnergy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int = 10,
    reference: str = "local",
) -> BalanceReport:
    """Average force scales across multiple batches.

    Args:
        model: TotalEnergy model.
        dataloader: DataLoader providing (R, seq) pairs.
        device: Torch device.
        n_batches: Number of batches to average over.
        reference: Reference term.

    Returns:
        BalanceReport with averaged scales.
    """
    model.eval()

    all_scales: Dict[str, List[float]] = {}

    for i, (R, seq, _, _) in enumerate(dataloader):
        if i >= n_batches:
            break

        R = R.to(device)
        seq = seq.to(device)

        forces = term_forces(model, R, seq)

        for k, F in forces.items():
            scale = _median_force_scale(F)
            all_scales.setdefault(k, []).append(scale)

    # Average scales
    avg_scales = {k: float(torch.tensor(v).mean()) for k, v in all_scales.items()}

    # Use reference term's scale to compute lambdas
    ref_scale = avg_scales.get(reference, 1.0)
    if ref_scale <= 0:
        raise ValueError(f"Reference term {reference} has zero scale")

    rec = {}
    for k, s in avg_scales.items():
        rec[k] = float(ref_scale / s)

    return BalanceReport(
        force_scales=avg_scales,
        recommended_lambdas=rec,
    )
