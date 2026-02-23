# src/calphaebm/data/__init__.py

"""Data loading and processing for PDB structures."""

from calphaebm.data.aa_map import aa3_to_idx, AA3_TO_AA1, AA1_TO_IDX
from calphaebm.data.pdb_parse import (
    download_cif,
    parse_cif_ca_chains,
    split_on_gaps,
    iter_fixed_segments,
    load_pdb_segments,
    get_residue_sequence,
)
from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.data.rcsb_query import (
    search_entries_xray_resolution,
    graphql_polymer_entities_for_entries,
    is_protein_only_entry,
    PolymerEntityInfo,
)
from calphaebm.data.build_pdb70_like import build_pdb70_like_polymer_entities
from calphaebm.data.synthetic import make_extended_chain, random_sequence

__all__ = [
    # Amino acid mapping
    "aa3_to_idx",
    "AA3_TO_AA1",
    "AA1_TO_IDX",
    
    # PDB parsing
    "download_cif",
    "parse_cif_ca_chains",
    "split_on_gaps",
    "iter_fixed_segments",
    "load_pdb_segments",
    "get_residue_sequence",
    
    # Dataset
    "PDBSegmentDataset",
    
    # RCSB API
    "search_entries_xray_resolution",
    "graphql_polymer_entities_for_entries",
    "is_protein_only_entry",
    "PolymerEntityInfo",
    
    # Dataset building
    "build_pdb70_like_polymer_entities",
    
    # Synthetic data
    "make_extended_chain",
    "random_sequence",
]