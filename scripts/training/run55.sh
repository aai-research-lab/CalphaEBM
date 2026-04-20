#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run55: No Contact Energy + Gentle α-Augmented DSM
#
#  Resume from run52 step 3500 (proven peak checkpoint).
#
#  Key changes vs run54:
#    - DISABLE E_contact: --disable-subterms contact
#      → E_contact = −λ·Σ h_i·h_j·g(r) is monotonically attractive
#      → sigmoid g(r) has no distance optimum — closer is always better
#      → this is likely the root cause of compaction (33% of energy,
#        overwhelms the 3% repulsion at distances above 4Å)
#    - E_geom (1601 params) already sees neighbor counts, mean distances,
#      and distance statistics — it can handle packing without E_contact
#    - Gentle α augmentation [0.85, 1.10] (same as run54)
#
#  Hypothesis: removing the monotonic attractive term eliminates
#  the compaction driving force. If 100K basin Rg holds >90%,
#  E_contact was the culprit.
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run55: No Contact + Gentle α-Augmented DSM"
echo "  Resume from: run52/step003500 (peak)"
echo "  DISABLED: E_contact (monotonic attractor)"
echo "  α ~ U(0.85, 1.10) — gentle 3× DSM"
echo "  Steps: 20000  |  LR: 6e-4 → 6e-5 cosine"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

calphaebm train --phase full \
    --pdb train_entities_9600.txt \
    --resume checkpoints/run52/run1/full/step003500.pt \
    --resume-model-only \
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
    `# ── Packing: E_Rg + E_coord (NO E_contact) ──` \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --coord-lambda 0.01 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --lambda-rg 0.0 \
    --disable-subterms contact \
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
    `# ── Balance (6 learned subterms now, r=7.0) ──` \
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
    --ckpt-dir checkpoints/run55 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run55.log
