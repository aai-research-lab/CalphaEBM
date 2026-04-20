"""B-factor calibration analysis for CalphaEBM.

Runs Langevin dynamics at multiple inverse temperatures (beta),
computes per-residue RMSF, converts to simulated B-factors
via B = 8pi^2 <u^2>, and compares to experimental B-factors from
X-ray crystallography.

Usage:
    calphaebm analyze bfactor \
        --checkpoint checkpoints/run43/run43_calibrated.pt \
        --pdb 1crn 1ubq 1pga 2ci2 \
        --betas 10 20 50 100 200
"""

# Lazy imports only — cli and core are imported on demand
# by the CLI dispatcher, not at package init time.
# This avoids circular imports when calphaebm.__init__ loads analysis.
