# tests/test_integration.py

"""Integration tests for CalphaEBM."""

import pytest

from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.energy import TotalEnergy
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import RepulsionEnergyLearnedRadius
from calphaebm.simulation.backends.pytorch import PyTorchSimulator
from calphaebm.simulation.io import TrajectorySaver
from calphaebm.utils.constants import EMB_DIM, NUM_AA


class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    @pytest.fixture
    def small_model(self, device):
        """Create small model for testing."""
        local = LocalEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM, hidden=(16, 16)).to(device)
        rep = RepulsionEnergyLearnedRadius(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)
        ss = SecondaryStructureEnergy(
            num_aa=NUM_AA, emb_dim=EMB_DIM, hidden=(16, 16)
        ).to(device)
        pack = PackingEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM, hidden_pair=(16, 16)).to(
            device
        )

        model = TotalEnergy(
            local=local,
            repulsion=rep,
            secondary=ss,
            packing=pack,
        ).to(device)

        return model

    def test_forward_backward(self, small_model, random_coords, random_seq):
        """Test forward and backward pass with all terms."""
        E = small_model(random_coords, random_seq).sum()
        E.backward()

        # Check that gradients exist
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_term_energies(self, small_model, random_coords, random_seq):
        """Test term_energies method with all terms."""
        terms = small_model.term_energies(random_coords, random_seq)
        assert "local" in terms
        assert "repulsion" in terms
        assert "secondary" in terms
        assert "packing" in terms


class TestSimulationPipeline:
    """Test end-to-end simulation pipeline."""

    def test_simulate_small(self, energy_model, random_coords, random_seq, device):
        """Test simulation with small model."""
        simulator = PyTorchSimulator(energy_model, device=device)

        result = simulator.run(
            R0=random_coords,
            seq=random_seq,
            n_steps=10,
            step_size=1e-4,
            log_every=5,
        )

        assert len(result.trajectories) > 0
        assert result.trajectories[0].shape == random_coords.shape

    def test_trajectory_saver(
        self, energy_model, random_coords, random_seq, device, tmp_path
    ):
        """Test trajectory saving."""
        # Run simulation
        simulator = PyTorchSimulator(energy_model, device=device)

        result = simulator.run(
            R0=random_coords,
            seq=random_seq,
            n_steps=20,
            step_size=1e-4,
            log_every=10,
        )

        # Save trajectory
        saver = TrajectorySaver(tmp_path)
        for frame in result.trajectories:
            saver.append(frame)

        metadata = {"test": True, "steps": 20}
        paths = saver.save_all(metadata)

        # Check files exist
        assert "npy" in paths
        assert "pt" in paths
        assert paths["npy"].exists()
        assert paths["pt"].exists()
