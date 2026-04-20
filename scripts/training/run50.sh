#!/usr/bin/env bash
set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "  Run50: E_Rg Flory term (λ=1.0), balanced packing"
echo "  Resume: checkpoints/run49/run1/full/step005000.pt (model only, fresh optimizer)"
echo "  Steps: 20000  |  LR: 6e-4→6e-5"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase full \
    --resume checkpoints/run49/run1/full/step005000.pt --resume-model-only \
    --pdb train_entities.no_test_entries.txt \
    --steps 20000 \
    --lr 6e-4 --lr-schedule cosine --lr-final 6e-5 \
    --scalar-lr-mult 1.0 \
    --sigma-min-rad 0.05 --sigma-max-rad 2.0 \
    --packing-rg-lambda 1.0 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --lambda-rg 0.0 \
    --elt-max-len 512 --elt-batch-size 16 --elt-every 2 \
    --lambda-funnel 0.5 --funnel-T 0.5 \
    --lambda-zscore 0.0 --target-zscore 3.0 \
    --lambda-gap 1.0 --gap-T 0.5 \
    --lambda-frustration 0.0 \
    --lambda-native-depth 0.5 --target-native-depth -1.0 \
    --lambda-balance 0.001 \
    --lambda-discrim 2.0 --discrim-every 2 --discrim-mode max \
    --lambda-basin 0.0 \
    --lambda-pack-contrastive 0.0 \
    --lambda-hb-beta-floor 0.1 \
    --langevin-beta 50.0 \
    --val-langevin-steps 500 \
    --val-max-samples 8 \
    --validate-every 500 \
    --ckpt-dir checkpoints/run50 \
    --ckpt-every 500 \
    2>&1 | tee logs/run50.log
