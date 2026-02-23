#!/usr/bin/env python
"""Quickstart example for CalphaEBM."""

import torch


def main():
    """Run quickstart example."""
    print("CalphaEBM Quickstart")
    print("=" * 50)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple local energy model
    from calphaebm.models.energy import TotalEnergy
    from calphaebm.models.local_terms import LocalEnergy

    local = LocalEnergy(num_aa=20, emb_dim=16, hidden=(128, 128))
    model = TotalEnergy(local=local).to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Generate synthetic data
    from calphaebm.data.synthetic import random_protein_like

    R, seq = random_protein_like(batch=2, length=50, device=device)
    print(f"Generated synthetic data: R shape {R.shape}, seq shape {seq.shape}")

    # Compute energy
    with torch.no_grad():
        E = model(R, seq)
        print(f"Energies: {E.cpu().numpy()}")

    # Run short simulation
    from calphaebm.simulation.backends.pytorch import PyTorchSimulator

    simulator = PyTorchSimulator(model, force_cap=50.0, device=device)

    result = simulator.run(
        R0=R[:1],  # Take first batch
        seq=seq[:1],
        n_steps=100,
        step_size=2e-4,
        log_every=20,
    )

    print(f"Simulation complete: {len(result.trajectories)} frames saved")
    print("Done!")


if __name__ == "__main__":
    main()
