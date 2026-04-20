"""Data loading utilities for parking (packing geometry calibration) analysis."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from tqdm import tqdm

from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class LoadStats:
    n_structures_loaded: int = 0
    n_structures_skipped: int = 0


def load_segments(
    segments_pt: Path,
    n_structures: Optional[int] = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[list, LoadStats]:
    """Load and optionally subsample structures from a processed segments .pt file.

    Args:
        segments_pt: Path to the segments file (produced by calphaebm train on first run).
                     Expected format: list of dicts with keys 'coords' (L,3) and 'seq' (L,).
        n_structures: If set, shuffle and subsample to this many structures.
        seed:         Random seed for reproducible subsampling.
        verbose:      Log progress.

    Returns:
        (data, stats) where data is a list of segment dicts.
    """
    stats = LoadStats()
    segments_pt = Path(segments_pt)

    if not segments_pt.exists():
        raise FileNotFoundError(
            f"Segments file not found: {segments_pt}\n"
            "This file is produced automatically during training. "
            "Run 'calphaebm train' at least once to generate it, or specify "
            "--segments explicitly."
        )

    logger.info("Loading segments from %s", segments_pt)
    data = torch.load(str(segments_pt), map_location="cpu", weights_only=False)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty list in {segments_pt}; got {type(data)}")

    # Validate keys
    required = {"coords"}
    sample = data[0]
    missing = required - set(sample.keys())
    if missing:
        raise ValueError(f"Segments missing required keys: {missing}. Got: {set(sample.keys())}")

    stats.n_structures_loaded = len(data)

    if n_structures and len(data) > n_structures:
        rng = random.Random(seed)
        rng.shuffle(data)
        stats.n_structures_skipped = len(data) - n_structures
        data = data[:n_structures]

    if verbose:
        logger.info(
            "Loaded %d structures (skipped %d)",
            len(data),
            stats.n_structures_skipped,
        )

    return data, stats


def segments_to_coord_batch(
    data: list,
    verbose: bool = True,
) -> torch.Tensor:
    """Pad segments to a (B, max_L, 3) float32 tensor.

    Structures shorter than max_L are zero-padded.
    """
    lengths = [d["coords"].shape[0] for d in data]
    max_L = max(lengths)
    B = len(data)

    R = torch.zeros(B, max_L, 3, dtype=torch.float32)
    for i, d in enumerate(data):
        coords = torch.as_tensor(d["coords"], dtype=torch.float32)
        L = coords.shape[0]
        R[i, :L] = coords

    if verbose:
        logger.info("Coord batch: shape=%s  max_L=%d", list(R.shape), max_L)

    return R
