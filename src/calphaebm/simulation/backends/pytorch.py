"""PyTorch-based Langevin dynamics simulator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from calphaebm.simulation.base import SimulationResult, Simulator
from calphaebm.simulation.observers import Observer
from calphaebm.utils.logging import get_logger
from calphaebm.utils.math import safe_norm

logger = get_logger()


class PyTorchSimulator(Simulator):
    """Overdamped Langevin simulator using PyTorch autograd.

    Implements:
        R_{t+1} = R_t + step_size * F(R_t) + sqrt(2 * step_size / beta) * noise

    with optional force clipping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 1.0,
        force_cap: Optional[float] = 50.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model)
        self.beta = beta
        self.force_cap = force_cap
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _compute_forces(self, R: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Compute forces = -grad(E)."""
        R.requires_grad_(True)
        E = self.model(R, seq).sum()
        F = -torch.autograd.grad(E, R, create_graph=False)[0]
        R.requires_grad_(False)
        return F

    def _clip_forces(self, F: torch.Tensor) -> tuple:
        """Clip forces to force_cap, return clipped forces and clip fraction."""
        if self.force_cap is None or self.force_cap <= 0:
            return F, 0.0

        norms = safe_norm(F, dim=-1, keepdim=True)
        clip_mask = norms > self.force_cap
        clip_frac = clip_mask.float().mean().item()

        scale = torch.clamp(self.force_cap / (norms + 1e-12), max=1.0)
        F_clipped = F * scale

        return F_clipped, clip_frac

    def run(
        self,
        R0: torch.Tensor,
        seq: torch.Tensor,
        n_steps: int,
        step_size: float,
        log_every: int = 50,
        **kwargs,
    ) -> SimulationResult:
        """Run Langevin dynamics.

        Args:
            R0: (B, L, 3) initial coordinates.
            seq: (B, L) amino acid indices.
            n_steps: Number of steps.
            step_size: Integration step size.
            log_every: Logging frequency.
            **kwargs: Additional arguments passed to observers.

        Returns:
            SimulationResult containing trajectory and metadata.
        """
        R = R0.to(self.device)
        seq = seq.to(self.device)

        # Clear observers and add trajectory observer if none present
        if not self.observers:
            from calphaebm.simulation.observers import TrajectoryObserver

            self.add_observer(TrajectoryObserver(save_every=log_every))

        # Reset all observers
        for obs in self.observers:
            obs.reset()

        # Noise scale from fluctuation-dissipation
        noise_scale = (2.0 * step_size / max(self.beta, 1e-12)) ** 0.5

        logger.info(f"Starting Langevin simulation for {n_steps} steps")
        logger.info(f"  step_size = {step_size:.2e}, beta = {self.beta}")
        logger.info(f"  force_cap = {self.force_cap}")

        # Initial frame - step passed as first argument, not in kwargs
        self._notify_observers(0, R, seq=seq, **kwargs)

        for step in range(1, n_steps + 1):
            # Compute forces
            F = self._compute_forces(R, seq)

            # Apply force clipping
            if self.force_cap:
                F, clip_frac = self._clip_forces(F)
            else:
                clip_frac = 0.0

            # Update positions
            noise = noise_scale * torch.randn_like(R)
            R = R + step_size * F + noise

            # Notify observers
            self._notify_observers(
                step,
                R,
                seq=seq,
                forces=F,
                force_cap=self.force_cap,
                clip_frac=clip_frac,
                **kwargs,
            )

            # Log progress
            if step % log_every == 0:
                with torch.no_grad():
                    E = self.model(R, seq).mean().item()
                    max_force = safe_norm(F, dim=-1).max().item()

                log_msg = (
                    f"step {step:6d}/{n_steps} | E={E:.3f} | max|F|={max_force:.3f}"
                )
                if self.force_cap:
                    log_msg += f" | clip={clip_frac:.3f}"
                logger.info(log_msg)

        # Gather results
        data = self._gather_observer_data()

        # Create result object
        result = SimulationResult(
            trajectories=data.get("trajectory", []),
            energies=data.get("total_energy"),
            min_distances=data.get("min_distance"),
            clip_fractions=data.get("clip_fraction"),
            metadata={
                "n_steps": n_steps,
                "step_size": step_size,
                "beta": self.beta,
                "force_cap": self.force_cap,
                "device": str(self.device),
                **kwargs,
            },
        )

        logger.info(f"Simulation complete: {len(result.trajectories)} frames saved")
        return result

    def run_with_snapshots(
        self,
        R0: torch.Tensor,
        seq: torch.Tensor,
        n_steps: int,
        step_size: float,
        save_every: int = 50,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Simplified interface returning only trajectory snapshots."""
        from calphaebm.simulation.observers import TrajectoryObserver

        self.clear_observers()
        self.add_observer(TrajectoryObserver(save_every=save_every))

        result = self.run(R0, seq, n_steps, step_size, **kwargs)

        return result.trajectories
