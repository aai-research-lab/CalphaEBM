#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════
#   Run59: Self-Consistent CG Training (Stage 2a)
#   Base: run57 step 3000 (Gaussian contact, no geom, E_delta=+0.229)
#
#   Loop: collect failures → retrain → evaluate → repeat
#   5 rounds × (100 proteins × 10K steps @ β=100 + 3K retrain)
#   Collection parallelized on 8 cores → ~15min/round collection
#   Relaxed thresholds: Rg<90%, RMSD>3, Q<0.8, Rg>115%
#
#   Goal: fix Rg 83% → 90%+ by learning from compaction failures
#   Key: --disable-subterms geom (keep geom=0 as in run57)
# ════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run59: Self-Consistent CG Training"
echo "  Base checkpoint: run57 step 3000 (E_delta=+0.229)"
echo "  Architecture: Gaussian contact + NO geom MLP"
echo "  5 rounds × 100 proteins × 10K steps @ β=100 (8 workers)"
echo "  Retrain: 3000 steps with DSM + contrastive"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

# macOS fork safety — required for parallel Langevin collection
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

calphaebm train \
    --phase self-consistent \
    --resume checkpoints/run57/run1/full/step003000_best.pt \
    --resume-model-only \
    --pdb train_entities_9600.txt \
    \
    --ckpt-dir checkpoints \
    --ckpt-prefix run59/run1 \
    \
    --disable-subterms geom \
    --max-rg-ratio 1.3 \
    \
    --n-rounds 5 \
    --collect-proteins 100 \
    --collect-steps 10000 \
    --collect-beta 100 \
    --collect-step-size 1e-4 \
    --collect-save-every 200 \
    --n-workers 8 \
    --collect-max-len 200 \
    \
    --retrain-steps 3000 \
    --retrain-lr 1e-4 \
    --lambda-sc 1.0 \
    --sc-margin 0.5 \
    --lambda-discrim 2.0 \
    --dsm-alpha-min 0.85 \
    --dsm-alpha-max 1.10 \
    \
    --sc-eval-steps 500 \
    --sc-eval-beta 50 \
    --sc-eval-proteins 8 \
    \
    --rg-compact 0.90 \
    --rg-swollen 1.15 \
    --q-false-basin 0.80 \
    --rmsd-drift 3.0 \
    --rmsf-frozen 0.3 \
    --ss-change-thr 0.3 \
    --max-negatives-per-protein 10 \
    \
    --convergence-threshold 0.05 \
    --min-negatives 10 \
    --sc-resume-round 1 \
    \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --coord-lambda 0.1 \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    \
    --cache-dir ./pdb_cache \
    --processed-cache-dir ./processed_cache \
    \
    2>&1 | tee logs/run59.log
