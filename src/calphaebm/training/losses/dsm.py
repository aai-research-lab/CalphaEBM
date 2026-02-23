# src/calphaebm/training/losses/dsm.py

"""Denoising Score Matching (DSM) loss for energy-based models.

The loss encourages the score (negative gradient) to point toward the
data manifold by corrupting data with Gaussian noise.
"""

import torch


def dsm_cartesian_loss(
    energy_model: torch.nn.Module,
    R: torch.Tensor,
    seq: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Denoising score matching loss in Cartesian space.
    
    Args:
        energy_model: Energy function that returns energy given (R, seq).
        R: (B, L, 3) clean coordinates.
        seq: (B, L) amino acid indices.
        sigma: Noise standard deviation (Å).
        
    Returns:
        Scalar loss.
    """
    # Add noise
    eps = torch.randn_like(R)
    R_tilde = R + sigma * eps
    R_tilde = R_tilde.detach().requires_grad_(True)
    
    # Compute energy and gradient
    E = energy_model(R_tilde, seq).sum()
    grad = torch.autograd.grad(E, R_tilde, create_graph=True)[0]
    
    # Target score: (R_tilde - R) / sigma^2
    target = (R_tilde - R) / (sigma ** 2)
    
    # MSE loss
    loss = ((grad - target) ** 2).mean()
    
    return loss


def dsm_internal_loss(
    energy_model: torch.nn.Module,
    l: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    seq: torch.Tensor,
    sigma_l: float = 0.1,
    sigma_theta: float = 0.1,
    sigma_phi: float = 0.2,
) -> torch.Tensor:
    """Denoising score matching loss in internal coordinate space.
    
    Args:
        energy_model: Energy function that accepts internal coordinates.
        l: (B, L-1) bond lengths.
        theta: (B, L-2) bond angles.
        phi: (B, L-3) torsion angles.
        seq: (B, L) amino acid indices.
        sigma_l, sigma_theta, sigma_phi: Noise scales.
        
    Returns:
        Scalar loss.
    """
    # Add noise to each internal coordinate
    eps_l = torch.randn_like(l)
    eps_theta = torch.randn_like(theta)
    eps_phi = torch.randn_like(phi)
    
    l_tilde = l + sigma_l * eps_l
    theta_tilde = theta + sigma_theta * eps_theta
    phi_tilde = phi + sigma_phi * eps_phi
    
    l_tilde = l_tilde.detach().requires_grad_(True)
    theta_tilde = theta_tilde.detach().requires_grad_(True)
    phi_tilde = phi_tilde.detach().requires_grad_(True)
    
    # Compute energy and gradients
    E = energy_model.energy_from_internals(l_tilde, theta_tilde, phi_tilde, seq).sum()
    
    grad_l = torch.autograd.grad(E, l_tilde, create_graph=True)[0]
    grad_theta = torch.autograd.grad(E, theta_tilde, create_graph=True)[0]
    grad_phi = torch.autograd.grad(E, phi_tilde, create_graph=True)[0]
    
    # Targets
    target_l = (l_tilde - l) / (sigma_l ** 2)
    target_theta = (theta_tilde - theta) / (sigma_theta ** 2)
    target_phi = (phi_tilde - phi) / (sigma_phi ** 2)
    
    # Combined loss
    loss = (
        ((grad_l - target_l) ** 2).mean() +
        ((grad_theta - target_theta) ** 2).mean() +
        ((grad_phi - target_phi) ** 2).mean()
    ) / 3.0
    
    return loss