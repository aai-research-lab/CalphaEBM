#!/bin/bash
#SBATCH --job-name=run1_eval100k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1_eval100k_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# 100k Langevin basin eval — the real test
# ============================================================
# Beta sweep: 100 (training), 500, 1000, 5000 (near-deterministic)
# 16 val proteins, L≤128, 100k steps each
# Watching: Rg drift, Q stability, compaction
# ============================================================

calphaebm evaluate \
    --mode basin \
    --checkpoint checkpoints/run1/run1/stage1_round010/step003000_best.pt \
    --pdb val_hq.txt \
    --langevin-beta 100 500 1000 5000 \
    --langevin-steps 100000 \
    --save-every 1000 \
    --log-every 10000 \
    --max-samples 16 \
    --max-len 128 \
    --no-early-stop \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --out-dir basin_results/run1_100k \
    --step-size 1e-4 \
    --force-cap 100.0
