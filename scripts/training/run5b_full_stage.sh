#!/bin/bash
#SBATCH --job-name=run5b_fs
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run5b_full_stage_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN5b: Warm-start from Run5 Round 5 best checkpoint
# Key fixes vs Run5:
#   - resume-model-only: load R5 R5 weights, reset round history
#   - val-beta: 100 → 50  (proper thermodynamic sampling)
#   - lambda-balance: 0.01 (keep default — balance must remain loud)

#   - scalar-lr-mult: 20 → 5  (main cause of lambda runaway)
#   - lambda-hb-beta-floor: 0.5  (prevent hb_beta collapse)
#   - max-rounds: 10 (fresh 10 rounds from R5 weights)
# ============================================================

echo "========================================"
echo "  RUN 5b: WARM START FROM R5 ROUND 5"
echo "  val-beta=50, scalar-lr-mult=5, hb-beta-floor=0.5"
echo "  Started: $(date)"
echo "========================================"

calphaebm train \
    --stage full \
    --resume checkpoints/run5/run5/stage1_round005/step002000_best.pt \
    --resume-model-only \
    --train-pdb train_hq.txt \
    --val-pdb val_hq.txt \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --ckpt-dir checkpoints/run5b \
    --ckpt-prefix run5b \
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
    --val-beta 50.0 \
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
    --scalar-lr-mult 5 \
    --lambda-hb-beta-floor 0.5

echo "========================================"
echo "  RUN 5b COMPLETE — $(date)"
echo "========================================"
