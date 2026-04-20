#!/bin/bash
#SBATCH --job-name=run9c_eval
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --output=/tmp/run9c_eval_%j.log
#SBATCH --error=/tmp/run9c_eval_%j.err

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

# Keep stdout/stderr writes small and unbuffered so no single write fills the pipe.
export PYTHONUNBUFFERED=1

# Force-offline every library that might try to call home on an airgapped HPC.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ============================================================
# RUN 9c EVAL WATCHER — independent of training job.
# Polls checkpoints/run9/run9/ for new round checkpoints.
# 16 proteins × 50K steps × 16 workers (1 protein per worker).
#
# Enriched metrics (via updated training_evaluation.py):
#   - Per-protein energy decomposition (local, rep, ss, pack)
#   - Rama basin populations (helix, sheet, PPII, turn, other)
#   - JS divergence vs PDB reference (rama, φφ, θθ, Δφ)
#   - Accept rate, RMSF, contact order, trajectory trends
#
# Logs to LOCAL /tmp with rsync --append mirror to NFS every 60s,
# matching the run9c training job infrastructure.
# NFS nfsd threads = 64 (server-side fix applied).
# ============================================================

# Background log mirror: /tmp -> NFS every 60s, append-only (rsync).
mkdir -p ~/CalphaEBM/logs
LOG_LOCAL=/tmp/run9c_eval_${SLURM_JOB_ID}.log
ERR_LOCAL=/tmp/run9c_eval_${SLURM_JOB_ID}.err
LOG_NFS=~/CalphaEBM/logs/run9c_eval_${SLURM_JOB_ID}.log
ERR_NFS=~/CalphaEBM/logs/run9c_eval_${SLURM_JOB_ID}.err

# Pre-create empty NFS destination files so rsync --append has something
# to extend rather than doing a full copy on the first pass.
: > "$LOG_NFS" 2>/dev/null
: > "$ERR_NFS" 2>/dev/null

(
    while true; do
        sleep 60
        # --append   transfers only bytes past destination file size
        # --inplace  appends in place (no rename-and-replace, tail -F friendly)
        # timeout 30 guards against NFS stalls
        timeout 30 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null &
        timeout 30 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null &
        wait
    done
) &
MIRROR_PID=$!

cleanup() {
    kill "$MIRROR_PID" 2>/dev/null
    # Final flush on exit with longer timeout.
    timeout 60 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null
    timeout 60 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null
}
trap cleanup EXIT

echo "========================================"
echo "  RUN 9c EVAL WATCHER"
echo "  16 proteins × 50K steps × β=100"
echo "  16 parallel workers"
echo "  Watching: checkpoints/run9/run9/"
echo "  Logs local: $LOG_LOCAL"
echo "  Mirrored to NFS every 60s (rsync --append): $LOG_NFS"
echo "  Started: $(date)"
echo "  Node: $(hostname)"
echo "========================================"

calphaebm evaluate --mode watch \
    --ckpt-dir    checkpoints/run9 \
    --ckpt-prefix run9 \
    --pdb         val_hq.txt \
    --cache-dir   pdb_cache \
    --n-samples   16 \
    --n-steps     50000 \
    --beta        100.0 \
    --sampler     mala \
    --max-rounds  50 \
    --poll-interval 60

echo "========================================"
echo "  RUN 9c EVAL WATCHER COMPLETE — $(date)"
echo "========================================"
