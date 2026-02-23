# src/calphaebm/simulation/factory.py

"""Factory function for creating simulators."""

from typing import Optional
import torch

from calphaebm.simulation.base import Simulator
from calphaebm.simulation.backends.pytorch import PyTorchSimulator


def create_simulator(
    backend: str = "pytorch",
    model: Optional[torch.nn.Module] = None,
    **kwargs,
) -> Simulator:
    """Create a simulator with the specified backend.
    
    Args:
        backend: Simulation backend ('pytorch' or 'openmm').
        model: Energy model (required for pytorch backend).
        **kwargs: Backend-specific arguments.
        
    Returns:
        Simulator instance.
        
    Raises:
        ValueError: If backend is unknown or model missing.
    """
    if backend == "pytorch":
        if model is None:
            raise ValueError("Model required for pytorch backend")
        return PyTorchSimulator(model, **kwargs)
    
    elif backend == "openmm":
        # Placeholder for future OpenMM integration
        raise NotImplementedError("OpenMM backend coming soon")
    
    else:
        raise ValueError(f"Unknown backend: {backend}")