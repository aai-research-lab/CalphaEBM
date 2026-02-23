# src/calphaebm/data/pdb_parse.py

"""Download and parse PDB/mmCIF files to extract Cα chains and segments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
from Bio.PDB import MMCIFParser

from calphaebm.data.aa_map import aa3_to_idx
from calphaebm.utils.logging import get_logger

logger = get_logger()

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


@dataclass
class ChainCA:
    """Cα trace of a single chain."""

    pdb_id: str
    chain_id: str
    coords: np.ndarray  # (L, 3) float32
    seq: np.ndarray  # (L,) int64 (amino acid indices)

    def __len__(self) -> int:
        return len(self.coords)


def download_cif(pdb_id: str, cache_dir: str, force: bool = False) -> str:
    """Download mmCIF file to cache directory.

    Args:
        pdb_id: 4-character PDB ID.
        cache_dir: Directory to store downloaded files.
        force: Force re-download even if file exists.

    Returns:
        Path to downloaded file.
    """
    pdb_id = pdb_id.lower()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id}.cif")

    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    logger.info(f"Downloading {pdb_id} from RCSB...")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        logger.debug(f"Downloaded {pdb_id} to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download {pdb_id}: {e}")
        raise

    return out_path


def parse_cif_ca_chains(cif_path: str, pdb_id: str) -> List[ChainCA]:
    """Parse mmCIF and extract Cα traces for each chain.

    Args:
        cif_path: Path to mmCIF file.
        pdb_id: PDB ID (for logging).

    Returns:
        List of ChainCA objects.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif_path)

    chains: List[ChainCA] = []
    model = next(structure.get_models())  # first model only

    for chain in model.get_chains():
        coords = []
        seq = []

        for res in chain.get_residues():
            # Skip waters/hetero unless standard AA
            resname = res.get_resname()
            aa_idx = aa3_to_idx(resname)
            if aa_idx is None:
                continue

            if "CA" not in res:
                continue

            ca = res["CA"].get_coord()
            coords.append(ca.astype(np.float32))
            seq.append(int(aa_idx))

        if len(coords) < 4:
            continue

        chains.append(
            ChainCA(
                pdb_id=pdb_id.lower(),
                chain_id=str(chain.id),
                coords=np.stack(coords, axis=0),
                seq=np.array(seq, dtype=np.int64),
            )
        )

        logger.debug(f"Found chain {chain.id} with {len(coords)} residues")

    return chains


def split_on_gaps(
    coords: np.ndarray,
    seq: np.ndarray,
    max_ca_jump: float = 4.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split chain at large Cα-Cα jumps (indicating missing residues).

    Args:
        coords: (L, 3) coordinates.
        seq: (L,) amino acid indices.
        max_ca_jump: Maximum allowed Cα-Cα distance (Å).

    Returns:
        List of (coords_seg, seq_seg) tuples.
    """
    assert coords.ndim == 2 and coords.shape[1] == 3
    if coords.shape[0] <= 1:
        return []

    # Compute consecutive distances
    diffs = coords[1:] - coords[:-1]
    d = np.sqrt((diffs * diffs).sum(axis=1))

    # Breakpoints where jump too large
    breaks = np.where(d > max_ca_jump)[0] + 1  # break after index

    segments = []
    start = 0
    for end in breaks:
        if end - start >= 4:  # minimum segment length
            segments.append((coords[start:end], seq[start:end]))
        start = end

    # Last segment
    if coords.shape[0] - start >= 4:
        segments.append((coords[start:], seq[start:]))

    return segments


def iter_fixed_segments(
    coords: np.ndarray,
    seq: np.ndarray,
    seg_len: int = 128,
    stride: int = 64,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield fixed-length windows from a contiguous segment.

    Args:
        coords: (L, 3) coordinates.
        seq: (L,) sequence.
        seg_len: Window length.
        stride: Stride between windows.

    Yields:
        (coords_window, seq_window) tuples.
    """
    L = coords.shape[0]
    if L < seg_len:
        return

    for start in range(0, L - seg_len + 1, stride):
        end = start + seg_len
        yield coords[start:end].copy(), seq[start:end].copy()


def load_pdb_segments(
    pdb_ids: List[str],
    cache_dir: str,
    seg_len: int = 128,
    stride: int = 64,
    max_ca_jump: float = 4.5,
    limit_segments: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Download and parse PDBs, returning fixed-length segments.

    Args:
        pdb_ids: List of PDB IDs.
        cache_dir: Directory for cached mmCIF files.
        seg_len: Segment length.
        stride: Stride for sliding window.
        max_ca_jump: Max allowed Cα-Cα distance.
        limit_segments: Max number of segments to return.

    Returns:
        List of segment dictionaries with keys:
            pdb_id, chain_id, coords, seq
    """
    segments: List[Dict[str, Any]] = []

    for pid in pdb_ids:
        try:
            cif_path = download_cif(pid, cache_dir=cache_dir)
            chains = parse_cif_ca_chains(cif_path, pdb_id=pid.lower())

            for chain in chains:
                # Split at gaps
                for cseg, sseg in split_on_gaps(chain.coords, chain.seq, max_ca_jump):
                    # Extract fixed windows
                    for cw, sw in iter_fixed_segments(cseg, sseg, seg_len, stride):
                        segments.append(
                            {
                                "pdb_id": chain.pdb_id,
                                "chain_id": chain.chain_id,
                                "coords": cw.astype(np.float32),
                                "seq": sw.astype(np.int64),
                            }
                        )

                        if limit_segments and len(segments) >= limit_segments:
                            return segments

        except Exception as e:
            logger.warning(f"Failed to process {pid}: {e}")
            continue

    return segments


def get_residue_sequence(
    pdb_id: str, cache_dir: str, chain_id: Optional[str] = None
) -> List[str]:
    """Get 1-letter amino acid sequence for a PDB chain.

    Args:
        pdb_id: PDB ID.
        cache_dir: Cache directory.
        chain_id: Specific chain (if None, use first chain).

    Returns:
        List of 1-letter codes.
    """
    from calphaebm.data.aa_map import idx_to_aa1

    cif_path = download_cif(pdb_id, cache_dir=cache_dir)
    chains = parse_cif_ca_chains(cif_path, pdb_id)

    if not chains:
        raise ValueError(f"No chains found for {pdb_id}")

    if chain_id:
        chain = next((c for c in chains if c.chain_id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found in {pdb_id}")
    else:
        chain = chains[0]
        logger.info(f"Using chain {chain.chain_id} for {pdb_id}")

    return [idx_to_aa1(idx) for idx in chain.seq]
