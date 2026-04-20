#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run57: Gaussian Contact + Strong Coordination — No E_geom
#
#  FROM SCRATCH. Radical simplification of packing energy:
#
#  DISABLED: E_geom (1601 params, monotonic geometry features)
#    → MLP learns compaction bias from asymmetric features
#    → 5 of 6 features respond monotonically to compaction
#    → Cannot learn bidirectional response from DSM alone
#
#  ENABLED: E_contact with GAUSSIAN g(r) (23 params)
#    → g(r) = exp(-(r - r_peak)² / (2σ²))
#    → r_peak=6.5Å, σ=1.5Å (learnable)
#    → Bell-shaped: penalizes BOTH too-close and too-far
#    → Replaces the old monotonic sigmoid that caused compaction
#
#  STRENGTHENED: E_coord λ=0.1 (was 0.01)
#    → Bidirectional coordination band penalty
#    → Per-AA [n_lo, n_hi] from PDB calibration
#    → Now 10× stronger — actual restoring force, not just safety net
#    → Miyazawa-Jernigan (1996) used exactly this approach:
#      "attractive contact pair term + unfavorable high packing density term"
#
#  Packing architecture:
#    E_packing = E_contact(Gaussian) + E_coord(λ=0.1) + E_Rg(λ=0.1)
#    23 learnable params + 0 analytical + 0 analytical = 23 params
#    (down from 1625 params)
#
#  REQUIRES: modified packing.py with Gaussian g(r) installed:
#    cp packing.py src/calphaebm/models/packing.py
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run57: Gaussian Contact + Strong Coord — No E_geom"
echo "  FROM SCRATCH (no resume)"
echo "  DISABLED: E_geom (monotonic MLP)"
echo "  ENABLED:  E_contact (Gaussian g(r), 23 params)"
echo "  STRONG:   E_coord λ=0.1 (10× previous)"
echo "  α ~ U(0.85, 1.10) — gentle 3× DSM"
echo "  Steps: 20000  |  LR: 6e-4 → 6e-5 cosine"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

calphaebm train --phase full \
    --pdb train_entities_9600.txt \
    --steps 20000 \
    --lr 6e-4 --lr-schedule cosine --lr-final 6e-5 \
    --scalar-lr-mult 1.0 \
    \
    `# ── DSM noise + gentle α augmentation ──` \
    --sigma-min-rad 0.05 --sigma-max-rad 2.0 \
    --dsm-alpha-min 0.85 --dsm-alpha-max 1.10 \
    \
    `# ── Dataset filters ──` \
    --max-rg-ratio 1.3 \
    --elt-max-len 512 --elt-batch-size 16 --elt-every 2 \
    \
    `# ── Packing: Gaussian E_contact + strong E_coord + E_Rg ──` \
    `# ── NO E_geom ──` \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --coord-lambda 0.1 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --lambda-rg 0.0 \
    --disable-subterms geom \
    \
    `# ── ELT losses ──` \
    --lambda-funnel 0.5 --funnel-T 0.5 \
    --lambda-zscore 0.0 --target-zscore 3.0 \
    --lambda-gap 1.0 --gap-T 0.5 \
    --lambda-frustration 0.0 \
    --lambda-native-depth 0.5 --target-native-depth -1.0 \
    \
    `# ── Per-subterm discrimination ──` \
    --lambda-discrim 2.0 --discrim-every 2 --discrim-mode max \
    \
    `# ── Balance (6 learned subterms, r=7.0) ──` \
    --lambda-balance 0.001 --balance-r 7.0 \
    \
    `# ── Lambda floors ──` \
    --lambda-hb-beta-floor 0.1 \
    \
    `# ── Disabled losses ──` \
    --lambda-basin 0.0 \
    --lambda-pack-contrastive 0.0 \
    \
    `# ── Validation ──` \
    --langevin-beta 50.0 \
    --val-langevin-steps 500 \
    --val-max-samples 8 \
    --validate-every 500 \
    \
    `# ── Checkpoints ──` \
    --ckpt-dir checkpoints/run57 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run57.log
