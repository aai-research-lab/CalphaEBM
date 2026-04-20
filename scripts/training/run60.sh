#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════
#   Run60: Self-Consistent CG Training (Stage 2a) — adjusted λ
#   Base: run57 step 3000 (Gaussian contact, no geom, E_delta=+0.229)
#
#   Changes vs run59:
#     λ_sc:      1.0 → 5.0   (SC contrastive drives the update)
#     λ_balance: n/a → 0.01  (prevent E_contact explosion)
#     lr:        1e-4 → 1e-5 (gentler fine-tuning)
#     resume:    round 2 retrain (1765 negatives, new losses)
#
#   Loop: collect failures → retrain (DSM + DSM-on-neg + contrastive) → eval
#   No α-augmentation — real negatives replace synthetic scaling
#   10 rounds × (100 proteins × 10K steps @ β=100 + 3K retrain)
#   Collection parallelized on 8 cores → ~15min/round collection
# ════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run60: Self-Consistent CG Training (adjusted λ)"
echo "  Base checkpoint: run57 step 3000 (E_delta=+0.229)"
echo "  Architecture: Gaussian contact + NO geom MLP"
echo "  Key changes vs run59:"
echo "    λ_sc:  1.0 → 5.0  (SC contrastive drives the update)"
echo "    λ_bal: n/a → 0.01 (prevent E_contact explosion)"
echo "    lr:    1e-4 → 1e-5 (gentler fine-tuning)"
echo "  Resumes from round 2 (1765 negatives) with new losses"
echo "  10 rounds × 100 proteins × 10K steps @ β=100 (8 workers)"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

# Round 1 (897 negs) and Round 2 (868 negs) already collected in run60
# --sc-resume-round 1 loads round001.pt model + both neg directories
# Delete round002.pt to force retraining with new losses (DSM-on-neg + Q-funnel)
if [ -f checkpoints/run60/run1/self-consistent/round002.pt ]; then
    rm -f checkpoints/run60/run1/self-consistent/round002.pt
    rm -f checkpoints/run60/run1/self-consistent/round002_best.pt
    echo "  Deleted round002.pt — will retrain with new losses"
fi

# macOS fork safety — required for parallel Langevin collection
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

calphaebm train \
    --phase self-consistent \
    --resume checkpoints/run60/run1/self-consistent/round001_best.pt \
    --resume-model-only \
    --pdb train_entities_9600.txt \
    \
    --ckpt-dir checkpoints \
    --ckpt-prefix run60/run1 \
    \
    --disable-subterms geom \
    --max-rg-ratio 1.3 \
    \
    --n-rounds 10 \
    --collect-proteins 100 \
    --collect-steps 10000 \
    --collect-beta 100 \
    --collect-step-size 1e-4 \
    --collect-save-every 200 \
    --n-workers 8 \
    --collect-max-len 200 \
    \
    --retrain-steps 3000 \
    --retrain-lr 1e-5 \
    --lambda-sampled-dsm 3.0 \
    --lambda-sampled-qf 1.5 \
    --lambda-sampled-gap 3.0 \
    --lambda-balance 0.01 \
    --sc-margin 0.5 \
    --lambda-discrim 2.0 \
    \
    --sc-eval-steps 10000 \
    --sc-eval-beta 100 \
    --sc-eval-proteins 8 \
    \
    --rg-compact 0.90 \
    --rg-swollen 1.10 \
    --q-false-basin 0.90 \
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
    2>&1 | tee logs/run60.log
