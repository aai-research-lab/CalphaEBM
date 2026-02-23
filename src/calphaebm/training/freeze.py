# src/calphaebm/training/freeze.py

"""Utilities for freezing/unfreezing modules during phased training."""

import torch.nn as nn
from typing import Optional, List, Union


def set_requires_grad(
    module: Optional[nn.Module],
    requires_grad: bool,
) -> None:
    """Set requires_grad for all parameters in a module.
    
    Args:
        module: PyTorch module (may be None).
        requires_grad: Whether parameters require gradients.
    """
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_module(module: Optional[nn.Module]) -> None:
    """Freeze all parameters in a module.
    
    Args:
        module: PyTorch module (may be None).
    """
    set_requires_grad(module, False)


def unfreeze_module(module: Optional[nn.Module]) -> None:
    """Unfreeze all parameters in a module.
    
    Args:
        module: PyTorch module (may be None).
    """
    set_requires_grad(module, True)


def freeze_by_names(
    model: nn.Module,
    names: List[str],
    freeze: bool = True,
) -> None:
    """Freeze/unfreeze modules by name.
    
    Args:
        model: Model containing submodules.
        names: List of submodule names (e.g., ['local', 'repulsion']).
        freeze: True to freeze, False to unfreeze.
    """
    for name in names:
        if hasattr(model, name):
            module = getattr(model, name)
            set_requires_grad(module, not freeze)
        else:
            raise ValueError(f"Model has no attribute '{name}'")


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all parameters that require gradients.
    
    Args:
        model: PyTorch model.
        
    Returns:
        List of trainable parameters.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_params(model: nn.Module) -> int:
    """Count number of trainable parameters.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)