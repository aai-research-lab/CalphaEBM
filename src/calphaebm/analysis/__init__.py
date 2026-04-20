"""Analysis tools for CalphaEBM.

This module provides analysis tools for:
- Repulsion analysis (RDF, enrichment)
- Backbone geometry analysis (theta, phi distributions)
- Packing analysis (packing geometry feature calibration)
- Coordination analysis (per-AA coordination statistics for packing energy)
- B-factor analysis (Langevin RMSF vs experimental B-factors)
"""

from . import repulsion
from . import backbone
from . import packing
from . import coordination
from . import hbonds
# bfactor is imported lazily by the CLI dispatcher (analyze.py)
# to avoid circular imports during package initialization.

__all__ = ['repulsion', 'backbone', 'packing', 'coordination', 'hbonds']
