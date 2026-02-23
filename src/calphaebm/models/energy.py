# src/calphaebm/models/energy.py

"""Composite energy combining all terms with learnable gates."""

import torch
import torch.nn as nn
from typing import Optional, Dict

from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.repulsion import RepulsionEnergy
from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.packing import PackingEnergy


class TotalEnergy(nn.Module):
    """Modular energy with explicit λ gates for all terms.
    
    E_total = λ_local * E_local + λ_rep * E_rep + λ_ss * E_ss + λ_pack * E_pack
    
    Args:
        local: LocalEnergy module.
        repulsion: Optional repulsion module.
        secondary: Optional secondary structure module.
        packing: Optional packing module.
        gate_local: Initial λ for local term.
        gate_rep: Initial λ for repulsion.
        gate_ss: Initial λ for secondary.
        gate_pack: Initial λ for packing.
    """
    
    def __init__(
        self,
        local: LocalEnergy,
        repulsion: Optional[RepulsionEnergy] = None,
        secondary: Optional[SecondaryStructureEnergy] = None,
        packing: Optional[PackingEnergy] = None,
        gate_local: float = 1.0,
        gate_rep: float = 1.0,
        gate_ss: float = 1.0,
        gate_pack: float = 1.0,
    ):
        super().__init__()
        self.local = local
        self.repulsion = repulsion
        self.secondary = secondary
        self.packing = packing
        
        # Register gates as buffers (not parameters)
        self.register_buffer("gate_local", torch.tensor(float(gate_local)))
        self.register_buffer("gate_rep", torch.tensor(float(gate_rep)))
        self.register_buffer("gate_ss", torch.tensor(float(gate_ss)))
        self.register_buffer("gate_pack", torch.tensor(float(gate_pack)))
        
    def forward(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute total energy.
        
        Args:
            R: (B, L, 3) coordinates.
            seq: (B, L) amino acid indices.
            
        Returns:
            (B,) Energy per batch element.
        """
        E = self.gate_local * self.local(R, seq)
        
        if self.repulsion is not None:
            E = E + self.gate_rep * self.repulsion(R, seq)
            
        if self.secondary is not None:
            E = E + self.gate_ss * self.secondary(R, seq)
            
        if self.packing is not None:
            E = E + self.gate_pack * self.packing(R, seq)
            
        return E
    
    @torch.no_grad()
    def term_energies(self, R: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute each term separately.
        
        Args:
            R: (B, L, 3) coordinates.
            seq: (B, L) amino acid indices.
            
        Returns:
            Dictionary mapping term names to energies (B,).
        """
        out = {"local": self.local(R, seq)}
        
        if self.repulsion is not None:
            out["repulsion"] = self.repulsion(R, seq)
            
        if self.secondary is not None:
            out["secondary"] = self.secondary(R, seq)
            
        if self.packing is not None:
            out["packing"] = self.packing(R, seq)
            
        return out
    
    @torch.no_grad()
    def set_gates(self, **kwargs) -> None:
        """Set gate values by name.
        
        Args:
            **kwargs: gate_local, gate_rep, gate_ss, gate_pack
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown gate: {k}")
            getattr(self, k).fill_(float(v))
            
    def get_gates(self) -> Dict[str, float]:
        """Get current gate values."""
        return {
            "local": float(self.gate_local.item()),
            "repulsion": float(self.gate_rep.item()),
            "secondary": float(self.gate_ss.item()),
            "packing": float(self.gate_pack.item()),
        }