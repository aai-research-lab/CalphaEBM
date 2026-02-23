# src/calphaebm/evaluation/io/loaders.py

"""Load coordinates from various formats."""

import os
import re
import glob
from typing import List, Optional
import numpy as np
import torch


def natural_key(p: str) -> int:
    """Extract numeric part for natural sorting."""
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 0


def load_coords_from_pt(path: str) -> np.ndarray:
    """Load coordinates from PyTorch snapshot file.
    
    Handles both raw tensors and dicts containing coordinates.
    
    Args:
        path: Path to .pt file.
        
    Returns:
        (N, 3) coordinates as float64 numpy array.
    """
    d = torch.load(path, map_location="cpu")
    
    # Case 1: raw tensor
    if isinstance(d, torch.Tensor):
        R = d
    # Case 2: dict
    elif isinstance(d, dict):
        for k in ("R", "coords", "xyz", "X", "pos", "positions"):
            if k in d:
                R = d[k]
                break
        else:
            raise KeyError(f"No coordinates key found in {path}")
    else:
        raise TypeError(f"Unexpected type: {type(d)}")
    
    # Normalize shape
    if R.ndim == 3:
        R = R[0]  # Take first batch
    
    if R.ndim != 2 or R.shape[-1] != 3:
        raise ValueError(f"Expected (N,3) shape, got {R.shape}")
    
    return R.float().cpu().numpy()


def load_coords_from_xyz(path: str) -> np.ndarray:
    """Load coordinates from XYZ file.
    
    Format:
        line1: N (number of atoms)
        line2: comment
        next N lines: element x y z
        
    Args:
        path: Path to .xyz file.
        
    Returns:
        (N, 3) coordinates as float64 numpy array.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    
    if len(lines) < 3:
        raise ValueError(f"File too short: {path}")
    
    try:
        n = int(lines[0].strip())
    except Exception as e:
        raise ValueError(f"First line not integer: {lines[0]}") from e
    
    body = lines[2:2 + n]
    if len(body) != n:
        raise ValueError(f"Expected {n} atoms, got {len(body)}")
    
    xyz = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed line: {line}")
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.asarray(xyz, dtype=np.float64)


def load_trajectory_from_dir(
    traj_dir: str,
    pattern: str = "snapshot_*.pt",
) -> List[np.ndarray]:
    """Load all snapshots from a directory."""
    paths = sorted(glob.glob(os.path.join(traj_dir, pattern)), key=natural_key)
    
    if not paths:
        raise ValueError(f"No files matching {pattern} in {traj_dir}")
    
    frames = []
    for p in paths:
        frames.append(load_coords_from_pt(p))
    
    return frames