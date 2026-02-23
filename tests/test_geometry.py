# tests/test_geometry.py

"""Tests for geometry module."""

import pytest
import torch
import numpy as np

from calphaebm.geometry.dihedral import dihedral, dihedral_from_points
from calphaebm.geometry.internal import (
    bond_lengths,
    bond_angles,
    torsions,
    check_geometry,
)
from calphaebm.geometry.features import phi_sincos


class TestDihedral:
    """Test dihedral angle calculations."""
    
    def test_dihedral_known_values(self):
        """Test against known dihedral values."""
        # Planar cis (0 degrees)
        p0 = torch.tensor([0.0, 0.0, 0.0])
        p1 = torch.tensor([1.0, 0.0, 0.0])
        p2 = torch.tensor([2.0, 0.0, 0.0])
        p3 = torch.tensor([3.0, 0.0, 0.0])
        phi = dihedral(p0, p1, p2, p3)
        assert abs(phi.item()) < 1e-6
        
        # Planar trans (180 degrees)
        p3 = torch.tensor([2.0, 1.0, 0.0])
        phi = dihedral(p0, p1, p2, p3)
        assert abs(abs(phi.item()) - np.pi) < 1e-6
        
        # Right angle (90 degrees)
        p3 = torch.tensor([2.0, 0.0, 1.0])
        phi = dihedral(p0, p1, p2, p3)
        assert abs(abs(phi.item()) - np.pi/2) < 1e-6
    
    def test_dihedral_periodicity(self):
        """Test that dihedral wraps correctly."""
        p0 = torch.tensor([0.0, 0.0, 0.0])
        p1 = torch.tensor([1.0, 0.0, 0.0])
        p2 = torch.tensor([2.0, 0.0, 0.0])
        
        # Rotate around
        angles = []
        for t in np.linspace(0, 2*np.pi, 10):
            p3 = torch.tensor([2.0, np.sin(t), np.cos(t)])
            angles.append(dihedral(p0, p1, p2, p3).item())
        
        # Should be continuous across 2π boundary
        angles = np.array(angles)
        assert np.all(angles >= -np.pi)
        assert np.all(angles <= np.pi)
    
    def test_dihedral_from_points(self):
        """Test batch dihedral calculation."""
        points = torch.randn(2, 10, 3)  # (B, L, 3)
        phi = dihedral_from_points(points)
        assert phi.shape == (2, 7)  # L-3


class TestInternalCoordinates:
    """Test internal coordinate calculations."""
    
    def test_bond_lengths(self, random_coords):
        """Test bond length calculation."""
        l = bond_lengths(random_coords)
        B, L, _ = random_coords.shape
        assert l.shape == (B, L-1)
        assert torch.all(l > 0)  # Lengths should be positive
    
    def test_bond_angles(self, random_coords):
        """Test bond angle calculation."""
        theta = bond_angles(random_coords)
        B, L, _ = random_coords.shape
        assert theta.shape == (B, L-2)
        assert torch.all(theta >= 0) and torch.all(theta <= np.pi)
    
    def test_torsions(self, random_coords):
        """Test torsion calculation."""
        phi = torsions(random_coords)
        B, L, _ = random_coords.shape
        assert phi.shape == (B, L-3)
        assert torch.all(phi >= -np.pi) and torch.all(phi <= np.pi)
    
    def test_check_geometry(self, simple_protein):
        """Test geometry validation."""
        result = check_geometry(simple_protein)
        assert "length" in result
        assert "bond_lengths" in result
        assert "min_nonbonded" in result
        assert "valid" in result


class TestFeatures:
    """Test feature transformations."""
    
    def test_phi_sincos(self):
        """Test sin/cos transformation."""
        phi = torch.tensor([0.0, np.pi/2, np.pi, -np.pi/2])
        sc = phi_sincos(phi)
        assert sc.shape == (4, 2)
        
        # Check values
        assert abs(sc[0, 0]) < 1e-6  # sin(0) = 0
        assert abs(sc[0, 1] - 1.0) < 1e-6  # cos(0) = 1
        
        assert abs(sc[1, 0] - 1.0) < 1e-6  # sin(π/2) = 1
        assert abs(sc[1, 1]) < 1e-6  # cos(π/2) = 0