# src/calphaebm/utils/seed.py

"""Reproducibility utilities for random seeds."""

import os
import random
import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")


def worker_init_fn(worker_id: int) -> None:
    """Initialize DataLoader workers with unique seeds.
    
    Use with torch.utils.data.DataLoader(worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)