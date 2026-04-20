#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run53: α-Augmented Bidirectional DSM
#
#  Resume from run52 step 3500 (peak checkpoint).
#
#  Key changes vs run52:
#    - α-AUGMENTED DSM: 3 samples per step, all pointing to native
#        Sample 1: x̃ → x_native       (standard IC noise)
#        Sample 2: x_α → x_native     (Rg scaling only)
#        Sample 3: x̃_α → x_native    (scaling + IC noise)
#      α ~ U(0.65, 1.25) — compacted and swollen structures
#      Fixes the coverage gap: DSM now sees both directions
#    - BALANCE: r=7.0, 7 learned subterms (coord/Rg excluded)
#    - DIAGNOSTICS: coord/Rg shown under "constraint" category
#    - ~3× slower per step due to 3 forward passes
#
#  Expected training time: ~50-60h on CPU (6-7min/50 steps × 20K)
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run53: α-Augmented Bidirectional DSM"
echo "  Resume from: run52/step003500 (peak)"
echo "  α ~ U(0.65, 1.25) — 3× DSM samples per step"
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
    `# ── DSM noise + α augmentation ──` \
    --sigma-min-rad 0.05 --sigma-max-rad 2.0 \
    --dsm-alpha-min 0.65 --dsm-alpha-max 1.25 \
    \
    `# ── Dataset filters ──` \
    --max-rg-ratio 1.3 \
    --elt-max-len 512 --elt-batch-size 16 --elt-every 2 \
    \
    `# ── Packing: E_Rg + E_coord ──` \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --coord-lambda 0.01 \
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
    --ckpt-dir checkpoints/run53 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run53.log
