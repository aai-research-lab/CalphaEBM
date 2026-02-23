# src/calphaebm/data/aa_map.py

"""Amino acid mapping utilities."""

from typing import Dict, Optional

# 3-letter to 1-letter mapping
AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # Common alternates
    "MSE": "M",  # selenomethionine -> methionine
    "SEC": "C",  # selenocysteine -> cysteine (approx)
    "PYL": "K",  # pyrrolysine -> lysine (approx)
    "HIP": "H",  # protonated histidine
    "HID": "H",  # histidine (delta protonated)
    "HIE": "H",  # histidine (epsilon protonated)
}

# 1-letter to index (0-19)
AA1_TO_IDX: Dict[str, int] = {
    aa: i for i, aa in enumerate(list("ACDEFGHIKLMNPQRSTVWY"))
}

# Reverse mapping: index to 1-letter
IDX_TO_AA1: Dict[int, str] = {i: aa for aa, i in AA1_TO_IDX.items()}

# Standard 3-letter order (for reference)
STANDARD_AA3: list = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def aa3_to_idx(resname3: str) -> Optional[int]:
    """Convert 3-letter amino acid code to index (0-19).

    Args:
        resname3: 3-letter code (case insensitive).

    Returns:
        Integer index 0-19, or None if not recognized.
    """
    if not resname3:
        return None

    resname3 = resname3.strip().upper()
    aa1 = AA3_TO_AA1.get(resname3)
    if aa1 is None:
        return None

    return AA1_TO_IDX.get(aa1)


def aa1_to_idx(aa1: str) -> Optional[int]:
    """Convert 1-letter amino acid code to index (0-19)."""
    return AA1_TO_IDX.get(aa1.upper())


def idx_to_aa1(idx: int) -> str:
    """Convert index (0-19) to 1-letter code."""
    return IDX_TO_AA1.get(idx, "X")


def get_all_aa_indices() -> list:
    """Return list of all standard amino acid indices (0-19)."""
    return list(range(20))
