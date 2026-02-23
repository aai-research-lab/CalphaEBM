# src/calphaebm/models/mlp.py

"""Simple MLP builder for energy terms."""

import torch.nn as nn
from typing import Tuple, Optional


def MLP(
    in_dim: int,
    hidden_dims: Tuple[int, ...] = (128, 128),
    out_dim: int = 1,
    dropout: float = 0.0,
    activation: str = "gelu",
) -> nn.Module:
    """Create a multilayer perceptron.
    
    Args:
        in_dim: Input dimension.
        hidden_dims: Hidden layer dimensions.
        out_dim: Output dimension.
        dropout: Dropout probability (0 = no dropout).
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh').
        
    Returns:
        nn.Sequential MLP.
    """
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    Act = acts.get(activation.lower(), nn.GELU)
    
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(Act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    
    return nn.Sequential(*layers)