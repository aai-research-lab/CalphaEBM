#!/bin/bash
#SBATCH --job-name=run9d_full
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/tmp/run9d_full_%j.log
#SBATCH --error=/tmp/run9d_full_%j.err

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

export PYTHONUNBUFFERED=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ============================================================
# RUN 9d: RESUME FROM ROUND 9 → complete R10
#
# Run9c (job 439) completed R6-R9 cleanly but hung at R10
# decoy-gen start. Root cause: ProcessPoolExecutor.map() pipe
# deadlock — 32768 tasks with 18-40KB pickled args each can
# fill the 64KB Linux pipe buffer during submission, blocking
# main in pipe_write while workers block in pipe_read.
#
# Fix applied in full_stage.py:
#   pool.map(_generate_one_decoy, tasks)
#   → pool.submit() + as_completed() with progress logging
#
# R9 checkpoint metrics (training-side):
#   loss=7.99  Q_af=8.5%  dRMSD_af=7.2%  E/res=-8.77
#   funnel slope=-94.8  hb_sheet.sigma1=1.66  rg_lambda=5.71
#
# This run completes the final round (R10/10) then exits.
# ============================================================

CKPT="checkpoints/run9/run9/full-stage/full_round009/step005000.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Round 9 checkpoint not found: $CKPT"
    exit 1
fi

mkdir -p ~/CalphaEBM/logs
LOG_LOCAL=/tmp/run9d_full_${SLURM_JOB_ID}.log
ERR_LOCAL=/tmp/run9d_full_${SLURM_JOB_ID}.err
LOG_NFS=~/CalphaEBM/logs/run9d_full_${SLURM_JOB_ID}.log
ERR_NFS=~/CalphaEBM/logs/run9d_full_${SLURM_JOB_ID}.err

: > "$LOG_NFS" 2>/dev/null
: > "$ERR_NFS" 2>/dev/null

(
    while true; do
        sleep 60
        timeout 30 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null &
        timeout 30 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null &
        wait
    done
) &
MIRROR_PID=$!

cleanup() {
    kill "$MIRROR_PID" 2>/dev/null
    timeout 60 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null
    timeout 60 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null
}
trap cleanup EXIT

echo "========================================"
echo "  RUN 9d: RESUME FROM ROUND 9 → R10"
echo "  Logs local: $LOG_LOCAL"
echo "  Mirrored to NFS: $LOG_NFS"
echo "  Fix: pool.map → submit+as_completed"
echo "  Resuming from: $CKPT"
echo "  Started: $(date)"
echo "  Node: $(hostname)"
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
echo "  RUN 9d COMPLETE — $(date)"
echo "========================================"
