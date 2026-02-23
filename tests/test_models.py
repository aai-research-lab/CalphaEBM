# tests/test_models.py

"""Tests for model modules."""

import torch

from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.embeddings import AAEmbedding
from calphaebm.models.mlp import MLP
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import (
    RepulsionEnergyFixed,
    RepulsionEnergyLearnedRadius,
)
from calphaebm.utils.constants import EMB_DIM, NUM_AA


class TestEmbeddings:
    """Test amino acid embeddings."""

    def test_embedding_shape(self, random_seq, device):
        """Test embedding output shape."""
        emb = AAEmbedding(num_aa=NUM_AA, dim=EMB_DIM).to(device)
        e = emb(random_seq)
        assert e.shape == (random_seq.shape[0], random_seq.shape[1], EMB_DIM)

    def test_embedding_values(self, random_seq, device):
        """Test embedding returns different values for different indices."""
        emb = AAEmbedding(num_aa=NUM_AA, dim=EMB_DIM).to(device)
        e = emb(random_seq)

        # Different positions in same sequence should have different embeddings
        if random_seq.shape[1] > 1:
            assert not torch.allclose(e[0, 0], e[0, 1])


class TestMLP:
    """Test MLP builder."""

    def test_mlp_shapes(self):
        """Test MLP output shape."""
        mlp = MLP(in_dim=10, hidden_dims=(20, 20), out_dim=5)
        x = torch.randn(3, 10)
        y = mlp(x)
        assert y.shape == (3, 5)

    def test_mlp_different_activations(self):
        """Test MLP with different activations."""
        for act in ["relu", "gelu", "silu", "tanh"]:
            mlp = MLP(in_dim=5, hidden_dims=(10,), out_dim=1, activation=act)
            x = torch.randn(2, 5)
            y = mlp(x)
            assert y.shape == (2, 1)


class TestLocalEnergy:
    """Test local energy term."""

    def test_forward_shape(self, local_model, random_coords, random_seq):
        """Test forward pass shape."""
        E = local_model(random_coords, random_seq)
        assert E.shape == (random_coords.shape[0],)

    def test_energy_from_internals(self, local_model, random_coords, random_seq):
        """Test energy from internal coordinates."""
        from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions

        length_idx = bond_lengths(random_coords)
        theta = bond_angles(random_coords)
        phi = torsions(random_coords)

        E1 = local_model(random_coords, random_seq)
        E2 = local_model.energy_from_internals(length_idx, theta, phi, random_seq)

        assert E1.shape == E2.shape
        # Values may differ slightly due to numerical precision
        assert torch.allclose(E1, E2, rtol=1e-4)

    def test_gradients(self, local_model, random_coords, random_seq):
        """Test that gradients can flow."""
        random_coords.requires_grad_(True)
        E = local_model(random_coords, random_seq).sum()
        E.backward()
        assert random_coords.grad is not None
        assert not torch.allclose(
            random_coords.grad, torch.zeros_like(random_coords.grad)
        )


class TestRepulsionEnergy:
    """Test repulsion energy terms."""

    def test_fixed_forward_shape(self, random_coords, device):
        """Test fixed repulsion forward pass."""
        rep = RepulsionEnergyFixed().to(device)
        E = rep(random_coords, None)
        assert E.shape == (random_coords.shape[0],)

    def test_learned_forward_shape(self, random_coords, random_seq, device):
        """Test learned repulsion forward pass."""
        rep = RepulsionEnergyLearnedRadius().to(device)
        E = rep(random_coords, random_seq)
        assert E.shape == (random_coords.shape[0],)

    def test_rho_bounds(self, random_seq, device):
        """Test learned radii stay within bounds."""
        rep = RepulsionEnergyLearnedRadius(
            rho_min=1.5,
            rho_max=3.0,
        ).to(device)

        rho = rep.rho(random_seq)
        assert torch.all(rho >= 1.5)
        assert torch.all(rho <= 3.0)

    def test_pair_r0_symmetric(self, random_seq, device):
        """Test pair radius is symmetric."""
        rep = RepulsionEnergyLearnedRadius().to(device)

        B, L = random_seq.shape
        # Create pairs
        seq_i = random_seq[:, :-1]
        seq_j = random_seq[:, 1:]

        r0_ij = rep.pair_r0(seq_i, seq_j)
        r0_ji = rep.pair_r0(seq_j, seq_i)

        assert torch.allclose(r0_ij, r0_ji)


class TestSecondaryEnergy:
    """Test secondary structure energy."""

    def test_forward_shape(self, random_coords, random_seq, device):
        """Test forward pass shape."""
        ss = SecondaryStructureEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)
        E = ss(random_coords, random_seq)
        assert E.shape == (random_coords.shape[0],)

    def test_energy_from_thetaphi(self, random_coords, random_seq, device):
        """Test energy from precomputed angles."""
        from calphaebm.geometry.internal import bond_angles, torsions

        ss = SecondaryStructureEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)

        theta = bond_angles(random_coords)
        phi = torsions(random_coords)

        E1 = ss(random_coords, random_seq)
        E2 = ss.energy_from_thetaphi(theta, phi, random_seq)

        assert E1.shape == E2.shape
        assert torch.allclose(E1, E2, rtol=1e-4)


class TestPackingEnergy:
    """Test packing energy."""

    def test_forward_shape(self, random_coords, random_seq, device):
        """Test forward pass shape."""
        pack = PackingEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)
        E = pack(random_coords, random_seq)
        assert E.shape == (random_coords.shape[0],)

    def test_energy_negative(self, random_coords, random_seq, device):
        """Test packing energy is attractive (negative)."""
        pack = PackingEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)
        E = pack(random_coords, random_seq)
        # Should be negative on average (attractive)
        assert E.mean().item() < 0


class TestTotalEnergy:
    """Test composite energy model."""

    def test_forward_shape(self, energy_model, random_coords, random_seq):
        """Test forward pass shape."""
        E = energy_model(random_coords, random_seq)
        assert E.shape == (random_coords.shape[0],)

    def test_term_energies(self, energy_model, random_coords, random_seq):
        """Test term_energies method."""
        terms = energy_model.term_energies(random_coords, random_seq)
        assert "local" in terms
        assert terms["local"].shape == (random_coords.shape[0],)

    def test_gates(self, energy_model):
        """Test gate setting/getting."""
        # Initial gates
        gates = energy_model.get_gates()
        assert "local" in gates

        # Set gates
        energy_model.set_gates(gate_local=2.0)
        assert energy_model.gate_local.item() == 2.0
