"""Tests for data module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from calphaebm.data.aa_map import AA1_TO_IDX, AA3_TO_AA1, aa3_to_idx
from calphaebm.data.synthetic import make_extended_chain, random_sequence


class TestAAMap:
    """Test amino acid mapping."""

    def test_aa3_to_idx_standard(self):
        """Test conversion of standard amino acids with correct indices."""
        # Test all standard amino acids with correct 0-based indices
        test_cases = [
            ("ALA", 0), ("CYS", 1), ("ASP", 2), ("GLU", 3), ("PHE", 4),
            ("GLY", 5), ("HIS", 6), ("ILE", 7), ("LYS", 8), ("LEU", 9),
            ("MET", 10), ("ASN", 11), ("PRO", 12), ("GLN", 13), ("ARG", 14),
            ("SER", 15), ("THR", 16), ("VAL", 17), ("TRP", 18), ("TYR", 19)
        ]

        for aa3, expected_idx in test_cases:
            assert aa3_to_idx(aa3) == expected_idx, f"{aa3} should map to {expected_idx}"

    def test_aa3_to_idx_case_insensitive(self):
        """Test case insensitivity."""
        assert aa3_to_idx("ala") == aa3_to_idx("ALA")

    def test_aa3_to_idx_unknown(self):
        """Test unknown amino acid."""
        assert aa3_to_idx("XXX") is None

    def test_aa3_to_idx_alternates(self):
        """Test alternate codes."""
        # MSE (selenomethionine) should map to M (index 10)
        assert aa3_to_idx("MSE") == 10


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
