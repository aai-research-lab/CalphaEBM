# src/calphaebm/training/phased.py

"""Phased training manager for curriculum learning."""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calphaebm.utils.logging import get_logger, ProgressBar
from calphaebm.training.freeze import freeze_module, unfreeze_module, set_requires_grad
from calphaebm.training.balancing import recommend_lambdas

logger = get_logger()


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    
    name: str  # 'local', 'repulsion', 'secondary', 'packing'
    terms: List[str]  # Which terms are active
    freeze: List[str]  # Which terms to freeze
    loss_fn: str  # 'dsm', 'contrastive', 'repulsion_calibrate'
    n_steps: int
    lr: float = 3e-4
    save_every: int = 500


@dataclass
class TrainingState:
    """Current training state."""
    
    step: int
    phase: str
    phase_step: int
    losses: Dict[str, float]
    gates: Dict[str, float]


class PhasedTrainer:
    """Manager for phased training of energy models.
    
    Handles:
    - Freezing/unfreezing terms appropriately
    - Checkpointing with experiment prefix
    - Loss function switching between phases
    - Progress logging and metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ckpt_dir: str = "checkpoints",
        experiment_prefix: str = "run1",
        save_mode: str = "full",  # 'full' or 'phase'
    ):
        self.model = model
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.experiment_prefix = experiment_prefix
        self.save_mode = save_mode
        
        self.optimizer = None
        self.current_phase = None
        self.current_step = 0
        self.phase_step = 0
        
        # Create checkpoint directory
        os.makedirs(ckpt_dir, exist_ok=True)
        
    def _phase_path(self, phase: str) -> str:
        """Get checkpoint directory for a phase."""
        return os.path.join(self.ckpt_dir, self.experiment_prefix, phase)
    
    def _checkpoint_path(self, phase: str, step: int) -> str:
        """Get checkpoint file path."""
        return os.path.join(self._phase_path(phase), f"step{step:06d}.pt")
    
    def save_checkpoint(self, phase: str, step: int, loss: float) -> str:
        """Save model checkpoint."""
        path = self._checkpoint_path(phase, step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        payload = {
            "step": step,
            "phase": phase,
            "loss": loss,
            "model_state": self.model.state_dict(),
            "gates": self.model.get_gates() if hasattr(self.model, "get_gates") else {},
        }
        
        if self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()
        
        torch.save(payload, path)
        logger.info(f"Saved checkpoint: {path}")
        return path
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        strict: bool = False,
    ) -> TrainingState:
        """Load checkpoint and return state."""
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model_state"], strict=strict)
        
        if load_optimizer and self.optimizer is not None and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        
        if "gates" in ckpt and hasattr(self.model, "set_gates"):
            self.model.set_gates(**ckpt["gates"])
        
        state = TrainingState(
            step=ckpt.get("step", 0),
            phase=ckpt.get("phase", "unknown"),
            phase_step=ckpt.get("step", 0),
            losses={"loss": ckpt.get("loss", 0.0)},
            gates=ckpt.get("gates", {}),
        )
        
        logger.info(f"Loaded checkpoint from {path} (step {state.step})")
        return state
    
    def find_latest_checkpoint(self, phase: str) -> Optional[str]:
        """Find latest checkpoint for a phase."""
        phase_dir = self._phase_path(phase)
        if not os.path.exists(phase_dir):
            return None
        
        checkpoints = [f for f in os.listdir(phase_dir) if f.startswith("step") and f.endswith(".pt")]
        if not checkpoints:
            return None
        
        # Sort by step number
        steps = [int(f.replace("step", "").replace(".pt", "")) for f in checkpoints]
        latest_idx = max(range(len(steps)), key=lambda i: steps[i])
        
        return os.path.join(phase_dir, checkpoints[latest_idx])
    
    def run_phase(
        self,
        config: PhaseConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume: Optional[str] = None,
    ) -> TrainingState:
        """Run a single training phase."""
        
        logger.info(f"Starting phase: {config.name}")
        logger.info(f"Active terms: {config.terms}")
        logger.info(f"Frozen terms: {config.freeze}")
        
        # Apply freezing
        for term in config.freeze:
            if hasattr(self.model, term):
                freeze_module(getattr(self.model, term))
                logger.debug(f"Froze {term}")
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(trainable_params)}")
        
        if len(trainable_params) > 0:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
        else:
            logger.warning("No trainable parameters in this phase")
            self.optimizer = None
        
        # Resume if requested
        start_step = 0
        if resume:
            if resume == "auto":
                resume = self.find_latest_checkpoint(config.name)
            
            if resume:
                state = self.load_checkpoint(resume, load_optimizer=True)
                start_step = state.step
                logger.info(f"Resumed from step {start_step}")
        
        # Training loop
        self.model.train()
        data_iter = iter(train_loader)
        
        progress = ProgressBar(config.n_steps, prefix=f"Phase {config.name}")
        
        for phase_step in range(start_step + 1, config.n_steps + 1):
            self.current_step += 1
            self.phase_step = phase_step
            
            # Get batch
            try:
                R, seq, _, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                R, seq, _, _ = next(data_iter)
            
            R = R.to(self.device)
            seq = seq.to(self.device)
            
            # Compute loss
            loss = self._compute_loss(config, R, seq)
            
            # Backward pass
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                self.optimizer.step()
            
            # Logging
            if phase_step == 1 or phase_step % 50 == 0:
                logger.info(
                    f"[{config.name}] step {phase_step:6d}/{config.n_steps} | "
                    f"loss={loss.item():.6f}"
                )
            
            # Save checkpoint
            if phase_step % config.save_every == 0:
                self.save_checkpoint(config.name, phase_step, loss.item())
            
            progress.update(1)
        
        # Final checkpoint
        final_path = self.save_checkpoint(config.name, config.n_steps, loss.item())
        logger.info(f"Phase {config.name} complete: {final_path}")
        
        return TrainingState(
            step=self.current_step,
            phase=config.name,
            phase_step=config.n_steps,
            losses={"loss": loss.item()},
            gates=self.model.get_gates() if hasattr(self.model, "get_gates") else {},
        )
    
    def _compute_loss(
        self,
        config: PhaseConfig,
        R: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for current phase."""
        
        if config.loss_fn == "dsm":
            from calphaebm.training.losses.dsm import dsm_cartesian_loss
            return dsm_cartesian_loss(self.model, R, seq, sigma=0.25)
        
        elif config.loss_fn == "contrastive":
            from calphaebm.training.losses.contrastive import contrastive_logistic_loss
            
            # This is a placeholder - actual contrastive loss depends on phase
            if config.name == "secondary":
                # Secondary structure: shuffle torsions
                from calphaebm.geometry.internal import bond_angles, torsions
                
                theta = bond_angles(R)
                phi = torsions(R)
                
                # Shuffle phi to break correlations
                B, nphi = phi.shape
                perm = torch.stack([torch.randperm(nphi, device=phi.device) for _ in range(B)], dim=0)
                phi_neg = torch.gather(phi, 1, perm)
                
                E_pos = self.model.secondary.energy_from_thetaphi(theta, phi, seq)
                E_neg = self.model.secondary.energy_from_thetaphi(theta, phi_neg, seq)
                
                return contrastive_logistic_loss(E_pos, E_neg)
            
            elif config.name == "packing":
                # Packing: use noisy structures as negatives
                from calphaebm.simulation.backends.pytorch import langevin_sample
                
                # Add noise and run short Langevin to get negatives
                R_noisy = R + 1.0 * torch.randn_like(R)
                snaps = langevin_sample(
                    self.model,
                    R0=R_noisy,
                    seq=seq,
                    n_steps=20,
                    step_size=2e-4,
                    force_cap=50.0,
                    log_every=1000,  # Disable logging
                )
                R_neg = snaps[-1].to(self.device)
                
                E_pos = self.model.packing(R, seq)
                E_neg = self.model.packing(R_neg, seq)
                
                return contrastive_logistic_loss(E_pos, E_neg)
            
            else:
                raise ValueError(f"Contrastive loss not implemented for phase {config.name}")
        
        elif config.loss_fn == "repulsion_calibrate":
            # Special case: repulsion calibration (no gradient)
            from calphaebm.utils.neighbors import pairwise_distances
            
            with torch.no_grad():
                # Compute minimum nonbonded distance
                B, L, _ = R.shape
                D = pairwise_distances(R)
                
                # Mask out bonded pairs
                for i in range(L):
                    D[:, i, max(0, i-2):min(L, i+3)] = float("inf")
                
                min_dist = D.amin(dim=(1, 2)).median()
                
                # Adjust gate_rep based on min distance
                current = self.model.gate_rep.item()
                target_min = 3.6  # Target minimum distance
                
                if min_dist < target_min:
                    new_gate = current * 1.1  # Increase repulsion
                else:
                    new_gate = current * 0.9  # Decrease repulsion
                
                new_gate = max(0.1, min(100.0, new_gate))
                self.model.set_gates(gate_rep=new_gate)
                
                # Return dummy loss
                return torch.tensor(min_dist, device=R.device, requires_grad=False)
        
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")