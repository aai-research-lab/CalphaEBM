#!/bin/bash
#SBATCH --job-name=run6_eval
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/eval_training_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

echo "========================================"
echo "  RUN 6: DETACHED BASIN EVAL WATCHER"
echo "  32 proteins x 10000 steps x beta=L  (poll every 60s)"
echo "  32 parallel workers (1 per protein)"
echo "  Started: $(date)"
echo "========================================"

calphaebm evaluate --mode watch \
    --ckpt-dir  checkpoints/run6 \
    --ckpt-prefix run6 \
    --pdb val_hq.txt \
    --cache-dir pdb_cache \
    --n-samples 32 \
    --n-steps   10000 \
    --beta      100.0 \
    --sampler   mala \
    --max-rounds 5 \
    --poll-interval 60   # seconds

echo "========================================"
echo "  EVAL WATCHER COMPLETE — $(date)"
echo "========================================"
