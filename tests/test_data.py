# tests/test_data.py

"""Tests for data module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from calphaebm.data.aa_map import aa3_to_idx, AA3_TO_AA1, AA1_TO_IDX
from calphaebm.data.synthetic import make_extended_chain, random_sequence


class TestAAMap:
    """Test amino acid mapping."""
    
    def test_aa3_to_idx_standard(self):
        """Test conversion of standard amino acids."""
        # Ala should map to 0
        assert aa3_to_idx("ALA") == 0
        # Gly should map to 6 (G is 6th in AC...)
        assert aa3_to_idx("GLY") == 6
    
    def test_aa3_to_idx_case_insensitive(self):
        """Test case insensitivity."""
        assert aa3_to_idx("ala") == aa3_to_idx("ALA")
    
    def test_aa3_to_idx_unknown(self):
        """Test unknown amino acid."""
        assert aa3_to_idx("XXX") is None
    
    def test_aa3_to_idx_alternates(self):
        """Test alternate codes."""
        # MSE (selenomethionine) should map to M
        assert aa3_to_idx("MSE") == AA1_TO_IDX["M"]


class TestSynthetic:
    """Test synthetic data generators."""
    
    def test_make_extended_chain_shape(self):
        """Test extended chain shape."""
        R = make_extended_chain(batch=3, length=10)
        assert R.shape == (3, 10, 3)
    
    def test_make_extended_chain_bond_length(self):
        """Test bond lengths in extended chain."""
        R = make_extended_chain(batch=1, length=5, bond=3.8, noise=0.0)
        # Should be along x-axis
        diffs = R[0, 1:] - R[0, :-1]
        lengths = np.sqrt((diffs ** 2).sum(axis=1))
        assert np.allclose(lengths, 3.8)
    
    def test_random_sequence(self):
        """Test random sequence generation."""
        seq = random_sequence(batch=2, length=10)
        assert seq.shape == (2, 10)
        assert seq.dtype == torch.long
        # Values should be in [0, 19]
        assert torch.all(seq >= 0)
        assert torch.all(seq < 20)