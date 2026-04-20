#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run52: Fresh training on curated Rg-filtered dataset
#
#  Key changes vs run50/51:
#    - NEW DATASET: train_entities_9600.txt (12,002 chains, Rg/Rg_Flory ≤ 1.3)
#      → No coiled-coils, fibrils, or extended helices
#      → 2.5× larger than run50 (12K vs 4.8K chains)
#      → X-ray + cryo-EM at ≤2.5Å resolution
#    - FRESH START: no checkpoint resume, learn from clean data
#    - E_coord band penalty (λ=0.01) as safety net
#    - E_Rg Flory (λ=1.0)
#    - --force-reprocess to build new Rg-filtered cache
#
#  Expected: ~9,600 train + ~2,400 val after 80/20 split
#  Expected training time: ~40h on CPU (2min/step × 20K steps)
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run52: Fresh training on Rg-filtered dataset"
echo "  Dataset: train_entities_9600.txt (12,002 chains)"
echo "  Steps: 20000  |  LR: 6e-4 → 6e-5 cosine"
echo "  No resume — training from scratch"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

calphaebm train --phase full \
    --pdb train_entities_9600.txt \
    --steps 20000 \
    --lr 6e-4 --lr-schedule cosine --lr-final 6e-5 \
    --scalar-lr-mult 1.0 \
    \
    `# ── DSM noise ──` \
    --sigma-min-rad 0.05 --sigma-max-rad 2.0 \
    \
    `# ── Dataset filters ──` \
    --max-rg-ratio 1.3 \
    --elt-max-len 512 --elt-batch-size 16 --elt-every 2 \
    --force-reprocess \
    \
    `# ── Packing: E_Rg + E_coord ──` \
    --packing-rg-lambda 1.0 \
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
    `# ── Balance ──` \
    --lambda-balance 0.001 \
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
    --ckpt-dir checkpoints/run52 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run52.log
