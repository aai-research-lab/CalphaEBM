# tests/conftest.py

"""Pytest fixtures for testing."""

import numpy as np
import pytest
import torch

from calphaebm.models.energy import TotalEnergy
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.utils.constants import EMB_DIM, NUM_AA


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Sequence length for tests."""
    return 10


@pytest.fixture
def random_coords(batch_size, seq_len, device):
    """Random coordinates for testing."""
    return torch.randn(batch_size, seq_len, 3, device=device)


@pytest.fixture
def random_seq(batch_size, seq_len, device):
    """Random sequence for testing."""
    return torch.randint(0, NUM_AA, (batch_size, seq_len), device=device)


@pytest.fixture
def local_model(device):
    """Local energy model."""
    return LocalEnergy(num_aa=NUM_AA, emb_dim=EMB_DIM).to(device)


@pytest.fixture
def energy_model(local_model, device):
    """Complete energy model with only local term."""
    return TotalEnergy(local=local_model).to(device)


@pytest.fixture
def simple_protein():
    """Simple protein-like coordinates (ideal helix)."""
    # Create a simple helix
    n = 10
    t = np.linspace(0, 4 * np.pi, n)
    coords = np.zeros((n, 3))
    coords[:, 0] = 2.3 * np.cos(t)  # x
    coords[:, 1] = 2.3 * np.sin(t)  # y
    coords[:, 2] = 1.5 * t / (2 * np.pi)  # z
    return coords.astype(np.float32)
