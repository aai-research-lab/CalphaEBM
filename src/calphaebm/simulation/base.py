# src/calphaebm/simulation/base.py

"""Base classes for simulation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    
    trajectories: List[torch.Tensor]  # List of frames (each B, N, 3)
    energies: Optional[List[float]] = None
    min_distances: Optional[List[float]] = None
    clip_fractions: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_frames(self) -> int:
        return len(self.trajectories)
    
    @property
    def n_atoms(self) -> int:
        return self.trajectories[0].shape[1]
    
    def get_coordinates_numpy(self) -> List:
        """Convert trajectories to numpy arrays."""
        return [t.cpu().numpy() for t in self.trajectories]


class Simulator(ABC):
    """Abstract base class for simulation backends."""
    
    def __init__(self, model: torch.nn.Module, **kwargs):
        self.model = model
        self.observers = []
        
    @abstractmethod
    def run(
        self,
        R0: torch.Tensor,
        seq: torch.Tensor,
        n_steps: int,
        step_size: float,
        **kwargs,
    ) -> SimulationResult:
        """Run simulation from initial coordinates."""
        pass
    
    def add_observer(self, observer):
        """Add an observer to collect data during simulation."""
        self.observers.append(observer)
    
    def clear_observers(self):
        """Remove all observers."""
        self.observers = []
    
    def _notify_observers(self, step: int, R: torch.Tensor, **kwargs):
        """Notify all observers of current state."""
        for obs in self.observers:
            obs.update(step, R, **kwargs)
    
    def _gather_observer_data(self) -> Dict[str, Any]:
        """Gather data from all observers."""
        data = {}
        for obs in self.observers:
            data.update(obs.get_results())
        return data