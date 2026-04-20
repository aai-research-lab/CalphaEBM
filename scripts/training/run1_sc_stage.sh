#!/bin/bash
#SBATCH --job-name=run1_sc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1_sc_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# RUN1 SC STAGE: Self-consistent training at a fixed β
# ============================================================
# Usage:
#   sbatch run1_sc_stage.sh                     # defaults: β=500, 10 rounds
#   sbatch run1_sc_stage.sh 100 10              # β=100, 10 rounds
#   sbatch run1_sc_stage.sh 50 10 /path/to.pt   # β=50, 10 rounds, custom ckpt
#
# On HPC (CUDA), the SC trainer exits after each round for fork
# safety. This script loops externally, re-launching each round.
# The trainer reads resume_next_round.txt to know where to continue.
# ============================================================

COLLECT_BETA=${1:-500}
N_ROUNDS=${2:-10}
INIT_CKPT=${3:-checkpoints/run1/run1/stage1_round010/step003000_best.pt}
SC_DIR=checkpoints/run1_sc/beta$(printf "%04d" $COLLECT_BETA)

echo "========================================"
echo "  SC STAGE: β=${COLLECT_BETA}, ${N_ROUNDS} rounds"
echo "  Initial checkpoint: ${INIT_CKPT}"
echo "  Output: ${SC_DIR}"
echo "========================================"

# Helper: read resume round from marker file
get_resume_round() {
    local marker="${SC_DIR}/run1/self-consistent/resume_next_round.txt"
    if [ -f "$marker" ]; then
        cat "$marker"
    else
        echo "0"
    fi
}

for ROUND in $(seq 1 $N_ROUNDS); do
    RESUME_ROUND=$(get_resume_round)
    echo ""
    echo "[β=${COLLECT_BETA}] Round ${ROUND}/${N_ROUNDS} (resume_round=${RESUME_ROUND}) — $(date)"
    echo ""

    calphaebm train \
        --stage sc \
        --resume "$INIT_CKPT" \
        --train-pdb train_hq.txt \
        --val-pdb val_hq.txt \
        --cache-dir pdb_cache \
        --processed-cache-dir processed_cache \
        --ckpt-dir $SC_DIR \
        --ckpt-prefix run1 \
        --energy-terms all \
        --n-rounds 1 \
        --collect-proteins 64 \
        --collect-steps 5000 \
        --collect-beta ${COLLECT_BETA}.0 \
        --collect-save-every 100 \
        --collect-max-len 512 \
        --n-workers 64 \
        --retrain-steps 3000 \
        --retrain-lr 1e-5 \
        --lambda-balance 0.01 \
        --lambda-discrim 2.0 \
        --lambda-sampled-hsm 1.0 \
        --lambda-sampled-qf 1.0 \
        --lambda-sampled-gap 1.0 \
        --sc-eval-steps 5000 \
        --sc-eval-beta 50.0 \
        --sc-eval-proteins 16 \
        --sc-resume-round $RESUME_ROUND \
        --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
        --backbone-data-dir analysis/backbone_geometry/data \
        --secondary-data-dir analysis/secondary_analysis/data \
        --repulsion-data-dir analysis/repulsion_analysis/data

    echo "[β=${COLLECT_BETA}] Round ${ROUND} done — $(date)"
done

echo "========================================"
echo "  SC STAGE β=${COLLECT_BETA} COMPLETE — $(date)"
echo "========================================"
