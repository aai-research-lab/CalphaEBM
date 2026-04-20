#!/bin/bash
#SBATCH --job-name=sc_curriculum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=logs/run1_sc_curriculum_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# ============================================================
# SC TEMPERATURE CURRICULUM: β = 50 → 100 → 200 → 500 → 1000
# ============================================================
# Hot→Cold: fix biggest landscape errors first, then refine
# Calls run1_sc_stage.sh for each beta, chaining checkpoints
# Total: 5 stages × 10 rounds × ~1hr/round ≈ 50 hours
# ============================================================

SCRIPT=scripts/training/run1_sc_stage.sh
N_ROUNDS=10
INIT_CKPT=checkpoints/run1/run1/stage1_round010/step003000_best.pt

# Helper: find best checkpoint in a stage directory
find_best() {
    local dir=$1/run1/self-consistent
    local best=$(ls -t ${dir}/*_best.pt 2>/dev/null | head -1)
    if [ -z "$best" ]; then
        best=$(ls -t ${dir}/round*.pt 2>/dev/null | head -1)
    fi
    echo "$best"
}

CKPT=$INIT_CKPT
for BETA in 50 100 200 500 1000; do
    echo "========================================"
    echo "  SC CURRICULUM: β=${BETA} (starting from $CKPT)"
    echo "  $(date)"
    echo "========================================"

    bash $SCRIPT $BETA $N_ROUNDS "$CKPT"

    # Find best checkpoint from this stage for next stage
    NEXT_CKPT=$(find_best checkpoints/run1_sc/beta$(printf "%04d" $BETA))
    if [ -n "$NEXT_CKPT" ]; then
        CKPT=$NEXT_CKPT
        echo "  Best from β=${BETA}: $CKPT"
    else
        echo "  WARNING: No checkpoint found for β=${BETA}, keeping previous"
    fi
done

echo "========================================"
echo "  SC CURRICULUM COMPLETE — $(date)"
echo "========================================"
