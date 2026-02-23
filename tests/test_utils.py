# tests/test_utils.py

"""Tests for utility modules."""

import pytest
import torch

from calphaebm.utils.math import safe_norm, wrap_to_pi, wrap_to_2pi, rmsd
from calphaebm.utils.neighbors import topk_nonbonded_pairs, NeighborList
from calphaebm.utils.smooth import smooth_switch, smooth_switch_derivative


class TestMath:
    """Test mathematical utilities."""
    
    def test_safe_norm(self):
        """Test safe norm calculation."""
        x = torch.tensor([[3.0, 4.0]])
        norm = safe_norm(x, dim=-1)
        assert abs(norm.item() - 5.0) < 1e-6
        
        # Test with zeros (should not produce NaN)
        x_zero = torch.zeros(2, 3)
        norm = safe_norm(x_zero, dim=-1)
        assert torch.all(torch.isfinite(norm))
    
    def test_wrap_to_pi(self):
        """Test angle wrapping to [-pi, pi)."""
        angles = torch.tensor([0.0, np.pi, 2*np.pi, -np.pi, -3*np.pi/2])
        wrapped = wrap_to_pi(angles)
        
        assert wrapped[0] == 0.0
        assert abs(wrapped[1] - np.pi) < 1e-6  # π stays π
        assert abs(wrapped[2]) < 1e-6  # 2π wraps to 0
        assert abs(wrapped[3] + np.pi) < 1e-6  # -π stays -π
        assert abs(wrapped[4] - np.pi/2) < 1e-6  # -3π/2 wraps to π/2
    
    def test_rmsd(self):
        """Test RMSD calculation."""
        P = np.random.randn(10, 3)
        Q = P.copy()  # Identical
        assert rmsd(P, Q) < 1e-6
        
        # Translated
        Q = P + 1.0
        assert rmsd(P, Q) < 1e-6  # Should be invariant


class TestNeighbors:
    """Test neighbor list utilities."""
    
    def test_topk_nonbonded_pairs_shape(self):
        """Test shape of neighbor pairs."""
        R = torch.randn(2, 10, 3)
        dist, idx = topk_nonbonded_pairs(R, K=5, exclude=2)
        
        assert dist.shape == (2, 10, 5)
        assert idx.shape == (2, 10, 5)
    
    def test_topk_exclude_bonded(self):
        """Test that bonded pairs are excluded."""
        R = torch.randn(1, 10, 3)
        dist, idx = topk_nonbonded_pairs(R, K=9, exclude=2)
        
        # Check that indices are not within exclude range
        for i in range(10):
            for j in idx[0, i]:
                if j < 10:
                    assert abs(i - j) > 2
    
    def test_neighbor_list_update(self):
        """Test neighbor list with skin."""
        nbl = NeighborList(cutoff=10.0, skin=2.0)
        R = torch.randn(1, 10, 3)
        
        # First update
        dist, idx = nbl.update(R)
        assert dist is not None
        
        # Small move (should not update)
        R_small = R + 0.1 * torch.randn_like(R)
        dist2, idx2 = nbl.update(R_small)
        assert dist2 is dist  # Same object (no update)
        
        # Large move (should update)
        R_large = R + 3.0 * torch.randn_like(R)
        dist3, idx3 = nbl.update(R_large)
        assert dist3 is not dist  # New object (update occurred)


class TestSmooth:
    """Test smooth switching functions."""
    
    def test_smooth_switch_values(self):
        """Test smooth switch values at boundaries."""
        r = torch.tensor([0.0, 5.0, 7.0, 9.0, 12.0])
        s = smooth_switch(r, r_on=6.0, r_cut=10.0)
        
        assert s[0] == 1.0  # r <= r_on
        assert s[1] == 1.0  # r <= r_on
        assert 0 < s[2] < 1  # in switching region
        assert 0 < s[3] < 1  # in switching region
        assert s[4] == 0.0  # r >= r_cut
    
    def test_smooth_switch_continuous(self):
        """Test smooth switch is continuous."""
        r_on, r_cut = 6.0, 10.0
        r = torch.linspace(5.0, 11.0, 100)
        s = smooth_switch(r, r_on, r_cut)
        
        # No large jumps
        diffs = torch.diff(s)
        assert torch.all(torch.abs(diffs) < 0.1)
    
    def test_smooth_switch_derivative(self):
        """Test derivative of smooth switch."""
        r = torch.tensor([5.0, 7.0, 8.0, 9.0, 11.0])
        ds = smooth_switch_derivative(r, r_on=6.0, r_cut=10.0)
        
        assert ds[0] == 0.0  # Outside region
        assert ds[1] < 0.0  # Decreasing
        assert ds[2] < 0.0  # Decreasing
        assert ds[3] < 0.0  # Decreasing
        assert ds[4] == 0.0  # Outside region