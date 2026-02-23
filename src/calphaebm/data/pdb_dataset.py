# src/calphaebm/data/pdb_dataset.py

"""PyTorch Dataset for PDB Cα segments."""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset

from calphaebm.data.pdb_parse import load_pdb_segments
from calphaebm.utils.logging import get_logger

logger = get_logger()


class PDBSegmentDataset(Dataset):
    """Dataset of fixed-length Cα segments from PDB structures.
    
    Args:
        pdb_ids: List of PDB IDs to load.
        cache_dir: Directory for cached mmCIF files.
        seg_len: Segment length (number of residues).
        stride: Stride for sliding window.
        max_ca_jump: Max allowed Cα-Cα distance for gap detection.
        limit_segments: Maximum number of segments to load.
        transform: Optional transform to apply to each segment.
        device: Device to place tensors on (None = CPU).
    """
    
    def __init__(
        self,
        pdb_ids: List[str],
        cache_dir: str = "./pdb_cache",
        seg_len: int = 128,
        stride: int = 64,
        max_ca_jump: float = 4.5,
        limit_segments: Optional[int] = None,
        transform: Optional[callable] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        self.transform = transform
        
        logger.info(f"Loading PDB segments from {len(pdb_ids)} IDs...")
        self.segments = load_pdb_segments(
            pdb_ids=pdb_ids,
            cache_dir=cache_dir,
            seg_len=seg_len,
            stride=stride,
            max_ca_jump=max_ca_jump,
            limit_segments=limit_segments,
        )
        
        logger.info(f"Loaded {len(self.segments)} segments")
        
        if len(self.segments) == 0:
            logger.warning("No segments loaded! Check PDB IDs and parameters.")
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Get a segment.
        
        Returns:
            (coords, seq, pdb_id, chain_id)
            - coords: (L, 3) float32 tensor
            - seq: (L,) int64 tensor of amino acid indices
            - pdb_id: string
            - chain_id: string
        """
        item = self.segments[idx]
        
        coords = torch.tensor(item["coords"], dtype=torch.float32)
        seq = torch.tensor(item["seq"], dtype=torch.long)
        
        if self.transform:
            coords, seq = self.transform(coords, seq)
        
        if self.device is not None:
            coords = coords.to(self.device)
            seq = seq.to(self.device)
        
        return coords, seq, item["pdb_id"], item["chain_id"]
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a segment without loading coordinates."""
        return {
            "pdb_id": self.segments[idx]["pdb_id"],
            "chain_id": self.segments[idx]["chain_id"],
            "length": len(self.segments[idx]["coords"]),
        }
    
    def filter_by_length(self, min_len: int, max_len: int) -> PDBSegmentDataset:
        """Create a new dataset filtered by segment length."""
        filtered = [s for s in self.segments if min_len <= len(s["coords"]) <= max_len]
        
        new_ds = PDBSegmentDataset.__new__(PDBSegmentDataset)
        new_ds.segments = filtered
        new_ds.device = self.device
        new_ds.transform = self.transform
        return new_ds