#!/bin/bash
#SBATCH --job-name=run1_fs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1_full_stage_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN1: Full-stage training (clean restart)
# ============================================================
# Changes from previous run1:
#   - Geom MLP removed from packing.py (was 1633 params, 64% of energy)
#   - Packing now = E_hp (23 params) + E_coord + E_rg
#   - --lambda-balance 0.01 (was 0.001) — 10x stronger
#   - Removed --disable-subterms geom (nothing to disable)
#   - Removed --packing-geom-calibration (no geom MLP)
#   - Balance r=6 (6 subterms), term r=4 (4 terms)
# ============================================================

calphaebm train \
    --stage full \
    --train-pdb train_hq.txt \
    --val-pdb val_hq.txt \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --ckpt-dir checkpoints/run1 \
    --ckpt-prefix run1 \
    --energy-terms all \
    --lr 5e-4 \
    --lr-final 5e-5 \
    --lr-schedule cosine \
    --max-rounds 10 \
    --steps-per-round 3000 \
    --n-decoys 8 \
    --lambda-native-depth 1.0 \
    --target-native-depth -1.0 \
    --lambda-balance 0.01 \
    --lambda-discrim 2.0 \
    --lambda-qf 1.0 \
    --lambda-rg 1.0 \
    --converge-q 0.95 \
    --converge-rmsd 5.0 \
    --converge-rg-lo 95.0 \
    --converge-rg-hi 105.0 \
    --val-proteins 16 \
    --val-steps 5000 \
    --val-beta 100.0 \
    --n-workers 16 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --backbone-data-dir analysis/backbone_geometry/data \
    --secondary-data-dir analysis/secondary_analysis/data \
    --repulsion-data-dir analysis/repulsion_analysis/data
