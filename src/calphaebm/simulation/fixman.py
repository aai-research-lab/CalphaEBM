# src/calphaebm/simulation/fixman.py
"""Fixman correction for IC-space sampling.

When sampling in internal coordinate (IC) space (θ, φ) rather than
Cartesian space R, the Jacobian of the NeRF map introduces a bias.
The Fixman correction restores detailed balance with respect to the
Cartesian Boltzmann distribution P(R) ∝ exp(-β·E(R)).

Background
----------
The NeRF reconstruction (θ, φ) → R is a nonlinear map. A uniform
distribution in (θ, φ) space does not correspond to a uniform
distribution in Cartesian space. The Jacobian of the map is:

    |J(θ, φ)| = ∏ᵢ sin(θᵢ)

(product over all bond angles; torsions φ contribute only a constant
factor that cancels in Metropolis ratios).

To sample the correct Boltzmann distribution P(R) ∝ exp(-β·E(R)) when
propagating in IC space, add the Fixman potential to the energy:

    U_Fixman(θ) = -(1/β) · Σᵢ log sin(θᵢ)

The corrected effective energy is:

    U_eff(θ, φ) = E(NeRF(θ, φ)) + U_Fixman(θ)
                = E(R) - (1/β) · Σᵢ log sin(θᵢ)

The gradient correction follows automatically via autograd — just add
U_Fixman to the energy before calling backward() or autograd.grad().

Reference
---------
Fixman, M. (1974). Classical statistical mechanics of constraints: a
theorem and application to polymers. Proc. Natl. Acad. Sci. USA, 71(8),
3050–3053.  https://doi.org/10.1073/pnas.71.8.3050

Notes
-----
- sin(θ) > 0 always because θ is clamped to (0.01, π − 0.01).
- The correction is purely θ-dependent; φ needs no correction.
- For near-native proteins (θ ≈ 110°–120°), sin(θ) ≈ 0.94–1.0 and
  the correction is small but non-zero.
- For large deformations or ab initio folding, the correction is
  significant and should not be omitted.
"""

from __future__ import annotations

import torch


def fixman_potential(theta: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute the Fixman correction potential.

    U_Fixman(θ) = -(1/β) · Σᵢ log sin(θᵢ)

    This is the scalar correction to be added to E(R) before computing
    gradients or Metropolis accept/reject ratios.

    Args:
        theta: Bond angle tensor, shape (1, L-2) or (L-2,).
               Must be clamped to (0, π) — i.e. sin(θ) > 0 guaranteed.
        beta:  Inverse temperature used for the sampler.

    Returns:
        Scalar tensor (0-dim) — the Fixman potential energy.
    """
    # sin(θ) ∈ (0, 1] since θ ∈ (0.01, π-0.01)
    # log sin(θ) ∈ (-∞, 0] — always negative or zero
    # -log sin(θ) ≥ 0 — penalises extreme bond angles (near 0 or π)
    return -(1.0 / beta) * torch.log(torch.sin(theta)).sum()


def fixman_log_jacobian(theta: torch.Tensor) -> torch.Tensor:
    """Compute log|J(θ)| = Σᵢ log sin(θᵢ).

    Useful for importance weighting trajectory snapshots or computing
    the exact Jacobian correction independent of β.

    Args:
        theta: Bond angle tensor, shape (1, L-2) or (L-2,).

    Returns:
        Scalar tensor — log of the Jacobian determinant.
    """
    return torch.log(torch.sin(theta)).sum()
