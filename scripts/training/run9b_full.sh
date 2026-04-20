#!/bin/bash
#SBATCH --job-name=run9b_full
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run9b_full_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

# ============================================================
# RUN 9b: RESUME FULL-STAGE FROM ROUND 1
#
# Uses --resume (NOT --resume-model-only) to preserve:
#   - Round counter (continues from round 2)
#   - Balance ramp state
#   - Optimizer state
#
# IMPORTANT: No --packing-*-lambda flags — these would override
# the learned buffer values back to defaults. The learnable
# buffers (rg_lambda=2.13, coord_lambda=0.43, etc.) are in
# the checkpoint and must be preserved.
#
# Fix applied: full_stage.py now calls zero_grad before fork.
# Fix applied: train_main.py uses .data.fill_() for buffer overrides.
# ============================================================

CKPT="checkpoints/run9/run9/full-stage/full_round001/step005000.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Round 1 checkpoint not found: $CKPT"
    exit 1
fi

echo "========================================"
echo "  RUN 9b: RESUME FROM ROUND 1"
echo "  --learn-all-buffers (~448 params)"
echo "  Preserving learned buffer values"
echo "  Resuming from: $CKPT"
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
    --collect-proteins 1024 \
    --elt-max-len 512 \
    --collect-max-len 512 \
    --learn-all-buffers \
    --resume "$CKPT"

echo "========================================"
echo "  RUN 9b FULL-STAGE COMPLETE — $(date)"
echo "========================================"
