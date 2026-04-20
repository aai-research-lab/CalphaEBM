#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run58: Full Bidirectional Packing — From Scratch
#
#  The complete fix: every packing component is bidirectional.
#
#  E_geom:    v2 features (7 bidirectional, per-AA centered)
#    → n_dev_signed: centered on per-AA mean (KEY: sign flips!)
#    → n_dev_abs: unsigned deviation magnitude
#    → shell_ratio: n_tight/n_medium (ratio, not count)
#    → shell_asymmetry: (n_tight - n_loose)/n_medium
#    → mean_r_dev: centered on calibrated global mean
#    → std_r_norm: kept from v1
#    → n_frac_band: position within [n_lo, n_hi] band
#    MLP: [seq_embed(16) ‖ geom(7)] → 32 → 16 → 1
#    1,634 params
#
#  E_contact: Gaussian g(r) = exp(-(r - r_peak)² / 2σ²)
#    → Peaks at r_peak≈6.5Å, penalizes both too-close and too-far
#    → 23 params (h vector + r_peak + σ + λ)
#
#  E_coord:   λ=0.1 (strong bidirectional coordination band)
#  E_Rg:      λ=0.1 (Flory scaling)
#
#  REQUIRES:
#    cp packing.py src/calphaebm/models/packing.py
#    cp model_builder.py src/calphaebm/cli/commands/train/model_builder.py
#    find src/ -type d -name __pycache__ -exec rm -rf {} +
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run58: Full Bidirectional Packing — From Scratch"
echo "  E_geom v2 (7 bidirectional features, 1634 params)"
echo "  E_contact (Gaussian g(r), 23 params)"
echo "  E_coord λ=0.1 | E_Rg λ=0.1"
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
    `# ── Packing: ALL subterms active ──` \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --coord-lambda 0.1 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --lambda-rg 0.0 \
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
    `# ── Balance (7 learned subterms, r=7.0) ──` \
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
    --ckpt-dir checkpoints/run58 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run58.log
