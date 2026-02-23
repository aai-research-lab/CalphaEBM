# tests/test_evaluation.py

"""Tests for evaluation module."""

import pytest
import numpy as np

from calphaebm.evaluation.metrics.rmsd import rmsd_kabsch, kabsch_rotate
from calphaebm.evaluation.metrics.contacts import (
    native_contact_set,
    q_hard,
    q_smooth,
    contact_count,
)
from calphaebm.evaluation.metrics.rdf import rdf_counts, rdf_normalized
from calphaebm.evaluation.metrics.rg import radius_of_gyration
from calphaebm.evaluation.metrics.clash import min_nonbonded, clash_probability


class TestRMSD:
    """Test RMSD calculations."""
    
    def test_rmsd_identical(self, simple_protein):
        """Test RMSD with identical structures."""
        rmsd = rmsd_kabsch(simple_protein, simple_protein)
        assert rmsd < 1e-6
    
    def test_rmsd_translation_invariant(self, simple_protein):
        """Test RMSD is translation invariant."""
        translated = simple_protein + np.array([10.0, 0.0, 0.0])
        rmsd1 = rmsd_kabsch(simple_protein, simple_protein)
        rmsd2 = rmsd_kabsch(simple_protein, translated)
        assert abs(rmsd1 - rmsd2) < 1e-6
    
    def test_kabsch_rotation(self, simple_protein):
        """Test Kabsch rotation alignment."""
        # Create rotated version
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotated = simple_protein @ R_mat.T
        
        aligned, R = kabsch_rotate(simple_protein, rotated)
        
        # Aligned should match rotated after centering
        rotated_centered = rotated - rotated.mean(axis=0)
        assert np.allclose(aligned, rotated_centered, atol=1e-5)


class TestContacts:
    """Test contact analysis."""
    
    def test_native_contact_set(self, simple_protein):
        """Test native contact set generation."""
        i, j, d0 = native_contact_set(simple_protein, cutoff=8.0, exclude=2)
        assert len(i) == len(j) == len(d0)
        assert all(d0 < 8.0)
    
    def test_q_hard(self, simple_protein):
        """Test Q_hard calculation."""
        i, j, d0 = native_contact_set(simple_protein, cutoff=8.0)
        
        # Self comparison should give Q=1
        q = q_hard(simple_protein, i, j, cutoff=8.0)
        assert abs(q - 1.0) < 1e-6
    
    def test_q_smooth(self, simple_protein):
        """Test Q_smooth calculation."""
        i, j, d0 = native_contact_set(simple_protein, cutoff=8.0)
        
        # Self comparison should give Q close to 1
        q = q_smooth(simple_protein, i, j, d0, beta=5.0, lam=1.0)
        assert q > 0.9
    
    def test_contact_count(self, simple_protein):
        """Test contact counting."""
        n_contacts = contact_count(simple_protein, cutoff=8.0, exclude=2)
        assert n_contacts >= 0
        assert isinstance(n_contacts, int)


class TestRDF:
    """Test RDF calculations."""
    
    def test_rdf_counts_shape(self, simple_protein):
        """Test RDF counts shape."""
        centers, counts = rdf_counts(simple_protein, r_max=10.0, dr=0.5)
        assert len(centers) == len(counts)
        assert len(centers) == 20  # 10.0 / 0.5
    
    def test_rdf_normalized(self, simple_protein):
        """Test RDF normalization."""
        centers, counts = rdf_counts(simple_protein, r_max=10.0, dr=0.5)
        g_norm, g_raw, tail_mean = rdf_normalized(centers, counts, dr=0.5)
        
        assert len(g_norm) == len(counts)
        assert tail_mean > 0


class TestRg:
    """Test radius of gyration."""
    
    def test_rg_calculation(self, simple_protein):
        """Test Rg calculation."""
        rg = radius_of_gyration(simple_protein)
        assert rg > 0
        
        # Linear chain along x
        linear = np.zeros((5, 3))
        linear[:, 0] = np.arange(5) * 3.8
        rg_linear = radius_of_gyration(linear)
        assert rg_linear > 0


class TestClash:
    """Test clash diagnostics."""
    
    def test_min_nonbonded(self, simple_protein):
        """Test minimum nonbonded distance."""
        med, mn = min_nonbonded(simple_protein, exclude=2)
        assert mn > 0
        assert med > 0
        assert mn <= med
    
    def test_clash_probability(self, simple_protein):
        """Test clash probability."""
        # Create trajectory with one clash
        traj = np.stack([simple_protein, simple_protein + 0.1])
        
        p_all, p_post = clash_probability(traj, threshold=3.0, burnin=1)
        assert 0 <= p_all <= 1
        assert 0 <= p_post <= 1