#!/bin/bash
#SBATCH --job-name=run5_fs
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run5_full_stage_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN5: Full-stage training (clean restart)
# ============================================================
# Changes from Run4:
#   - PackingEnergy v5: 22 params (was 21)
#   - Best et al. sigmoid: r_half=8.0, tau=0.2 (was 7.0/1.0)
#   - Contact density ρ energy term (new, λ=0.1)
#   - Exponential constraint penalties (was quadratic)
#   - Rg penalty: dead zone ±30%, saturating (was quadratic)
#   - Q-scaled saturating margins: m=5, α=5 (was constant 0.5)
#   - Sampler: mala (was langevin)
#   - IC noise: σ ~ [π/60, π/3] (was [0.05, 1.5])
#   - Single source of truth: defaults.py
#   - Config saved in checkpoints
#   - coord_n_star.json recomputed with Best-style sigmoid
#   - backbone analysis recomputed on clean 2280 monomers
#   - rg_lambda=1.0 (was 10.0), coord_lambda=0.1 (was 1.0)
#   - depth target=-3.0 (was -1.0)
# ============================================================

echo "========================================"
echo "  RUN 5: FULL-STAGE TRAINING (FROM SCRATCH)"
echo "  PackingEnergy v5 + Best sigmoid + ρ term"
echo "  Saturating exponential margins (m=5, α=5)"
echo "  Sampler: MALA"
echo "  Started: $(date)"
echo "========================================"

calphaebm train \
    --stage full \
    --train-pdb train_hq.txt \
    --val-pdb val_hq.txt \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --ckpt-dir checkpoints/run5 \
    --ckpt-prefix run5 \
    --energy-terms all \
    --local-window-size 8 \
    --lr 5e-4 \
    --lr-final 5e-5 \
    --lr-schedule cosine \
    --max-rounds 10 \
    --steps-per-round 2000 \
    --n-decoys 8 \
    --lambda-native-depth 2.0 \
    --target-native-depth -3.0 \
    --lambda-balance 0.01 \
    --lambda-discrim 2.0 \
    --lambda-qf 1.0 \
    --lambda-rg 2.0 \
    --lambda-gap 3.0 \
    --val-proteins 64 \
    --val-steps 10000 \
    --val-beta 100.0 \
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
    --collect-max-len 512

echo "========================================"
echo "  RUN 5 FULL-STAGE COMPLETE — $(date)"
echo "========================================"
