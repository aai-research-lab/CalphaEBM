#!/bin/bash
#SBATCH --job-name=run9_eval
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run9_eval_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN 9 EVAL WATCHER — independent of SC training job.
# Polls checkpoints/run9/run9/ for new round checkpoints.
# 16 proteins × 50K steps × 16 workers (1 protein per worker).
#
# Enriched metrics (via updated training_evaluation.py):
#   - Per-protein energy decomposition (local, rep, ss, pack)
#   - Rama basin populations (helix, sheet, PPII, turn, other)
#   - JS divergence vs PDB reference (rama, φφ, θθ, Δφ)
#   - Accept rate, RMSF, contact order, trajectory trends
# ============================================================

echo "========================================"
echo "  RUN 9 EVAL WATCHER"
echo "  16 proteins × 50K steps × β=L"
echo "  16 parallel workers"
echo "  Watching: checkpoints/run9/run9/"
echo "  Enriched metrics: JS divergence, energy decomposition"
echo "  Started: $(date)"
echo "========================================"

calphaebm evaluate --mode watch \
    --ckpt-dir    checkpoints/run9 \
    --ckpt-prefix run9 \
    --pdb         val_hq.txt \
    --cache-dir   pdb_cache \
    --n-samples   16 \
    --n-steps     50000 \
    --beta        100.0 \
    --sampler     mala \
    --max-rounds  50 \
    --poll-interval 60

echo "========================================"
echo "  RUN 9 EVAL WATCHER COMPLETE — $(date)"
echo "========================================"
