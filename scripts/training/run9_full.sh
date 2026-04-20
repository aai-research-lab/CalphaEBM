#!/bin/bash
#SBATCH --job-name=run9_full
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run9_full_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

# ============================================================
# RUN 9: FULL-STAGE TRAINING FROM SCRATCH — LEARNABLE BUFFERS
#
# Key changes from Run8:
#   1. --learn-all-buffers: ALL fixed PDB-derived buffer params
#      become learnable (~462 new params), initialized from
#      current PDB statistics:
#        - packing coords: n_star, sigma        [200 params]
#        - packing density: rho_fit_a/b/c, σ_ρ  [20 params]
#        - penalty shapes: dead_zone, m, alpha   [9 params]
#        - packing bounds: coord/rho lo/hi       [210 params]
#        - penalty strengths: rg/coord/rho λ     [3 params]
#        - gate geometry: rama basin centers/σ    [10 params]
#        - hbond geometry: mu/sigma distances     [6 params]
#
#   2. Full-stage (not SC): PDB-only training with IC-noised
#      decoys. Faster iteration, cleaner signal for buffer
#      learning before moving to SC in Run10.
#
# Motivation:
#   - Physics prior sweep: fixed buffers create non-native
#     minima 7× deeper than native (1PGB E=-2.7 vs E_nat=-0.4)
#   - Run8 R30: 86% discrimination from fixed geometry terms,
#     only 4% from packing. hb_α frozen. θφ anti-discriminating.
#   - Learnable buffers let training reshape penalty wells.
#
# Architecture:
#   PackingEnergy v6 + 5-group product Gaussian
#   dRMSD-funnel (topology-sensitive)
#   Saturating exponential margins (m=5, α=5)
#   MALA sampler
#
# Training:
#   10 rounds × 5000 steps/round = 50K gradient steps
#   lr: 5e-4 → 5e-5 cosine
#   32 IC-noised decoys per PDB structure
#   48 workers for decoy generation
#
# Losses:
#   depth=2.0 | discrim=2.0 | balance=0.01
#   qf=0.1 | drmsd=0.1
#
# Checkpoints: checkpoints/run9/run9/stage1_round{N:03d}/
# Eval: run9_eval_watcher.sh as separate SLURM job.
# ============================================================

echo "========================================"
echo "  RUN 9: FULL-STAGE — LEARNABLE BUFFERS"
echo "  --learn-all-buffers (~462 new params)"
echo "  PackingEnergy v6 + dRMSD-funnel"
echo "  10 rounds × 5000 steps | 48 workers"
echo "  lr: 5e-4 → 5e-5 cosine"
echo "  Training from scratch"
echo "  Started: $(date)"
echo "========================================"

calphaebm train \
    --stage full \
    --train-pdb train_hq.txt \
    --val-pdb val_hq.txt \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --ckpt-dir checkpoints/run9 \
    --ckpt-prefix run9 \
    --energy-terms all \
    --local-window-size 8 \
    --lr 5e-4 \
    --lr-final 5e-5 \
    --lr-schedule cosine \
    --max-rounds 10 \
    --steps-per-round 5000 \
    --n-decoys 32 \
    --lambda-native-depth 2.0 \
    --target-native-depth -3.0 \
    --lambda-balance 0.01 \
    --lambda-discrim 2.0 \
    --lambda-qf 0.1 \
    --lambda-drmsd 0.1 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --backbone-data-dir analysis/backbone_geometry/data \
    --secondary-data-dir analysis/secondary_analysis/data \
    --repulsion-data-dir analysis/repulsion_analysis/data \
    --packing-rg-lambda 1.0 \
    --packing-rho-lambda 1.0 \
    --hp-penalty-lambda 1.0 \
    --rho-penalty-lambda 1.0 \
    --collect-proteins 1024 \
    --elt-max-len 512 \
    --collect-max-len 512 \
    --learn-all-buffers

echo "========================================"
echo "  RUN 9 FULL-STAGE COMPLETE — $(date)"
echo "========================================"
