#!/bin/bash
#SBATCH --job-name=run7_eval
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run7_eval_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN 7 EVAL WATCHER — independent of SC training job.
# Polls checkpoints/run7/run7/ for new round checkpoints.
# 16 proteins x 50K steps x 16 workers (1 protein per worker).
# ============================================================

echo "========================================"
echo "  RUN 7 EVAL WATCHER"
echo "  16 proteins x 50K steps x beta=100"
echo "  16 parallel workers"
echo "  Watching: checkpoints/run7/run7/"
echo "  Started: $(date)"
echo "========================================"

calphaebm evaluate --mode watch \
    --ckpt-dir    checkpoints/run7 \
    --ckpt-prefix run7 \
    --pdb         val_hq.txt \
    --cache-dir   pdb_cache \
    --n-samples   16 \
    --n-steps     50000 \
    --beta        100.0 \
    --sampler     mala \
    --max-rounds  50 \
    --poll-interval 60

echo "========================================"
echo "  RUN 7 EVAL WATCHER COMPLETE — $(date)"
echo "========================================"
