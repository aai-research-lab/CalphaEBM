#!/bin/bash
#SBATCH --job-name=run9c_full
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/tmp/run9c_full_%j.log
#SBATCH --error=/tmp/run9c_full_%j.err

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

# Keep stdout/stderr writes small and unbuffered so no single write fills the pipe.
export PYTHONUNBUFFERED=1

# Force-offline every library that might try to call home on an airgapped HPC.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ============================================================
# RUN 9c: RESUME FROM ROUND 5
#
# Previous run (job 438) hung at the start of round 6 with an
# NFS write stall on the SLURM log file descriptor. Diagnosis:
#   - /home is NFS-mounted with `hard` option
#   - SLURM log was being written to /home/.../logs/...
#   - NFS server had a transient stall; `hard` means infinite block
#   - main thread stuck in pipe_write -> frozen training
#
# This run:
#   - logs to LOCAL /tmp on the compute node
#   - mirrors to NFS every 60s in the background using rsync --append
#     (appends only new bytes -> tail -F on NFS is flicker-free)
#   - disables all network-facing libraries (WandB, HF, etc.)
#   - resumes from Round 5 (last completed round)
#
# NFS server-side fix also applied: nfsd threads bumped from 8 -> 64
# (both /etc/nfs.conf and /proc/fs/nfsd/threads). Decoy generation
# dropped from ~300s to ~100s because 48 workers no longer serialize
# through the 8-thread server bottleneck on PDB cache reads.
#
# The calphaebm training code and the --learn-all-buffers flags
# are unchanged from run9b. R5 is a valid checkpoint with:
#   loss=8.21  Q_af=8.1%  dRMSD_af=6.5%  E/res=-8.37
#   funnel slope=-79.2  hb_sheet.sigma1=1.67  rg_lambda=4.72
# ============================================================

CKPT="checkpoints/run9/run9/full-stage/full_round005/step005000.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Round 5 checkpoint not found: $CKPT"
    exit 1
fi

# Background log mirror: /tmp -> NFS every 60s, append-only (rsync).
# Using rsync --append --inplace means only new bytes are appended in
# place on the destination; the NFS file grows monotonically, so
# `tail -F` on it never sees truncation or file-replacement.
mkdir -p ~/CalphaEBM/logs
LOG_LOCAL=/tmp/run9c_full_${SLURM_JOB_ID}.log
ERR_LOCAL=/tmp/run9c_full_${SLURM_JOB_ID}.err
LOG_NFS=~/CalphaEBM/logs/run9c_full_${SLURM_JOB_ID}.log
ERR_NFS=~/CalphaEBM/logs/run9c_full_${SLURM_JOB_ID}.err

# Pre-create the NFS destination files empty so rsync --append has
# something to extend. Otherwise the first rsync does a full copy.
: > "$LOG_NFS" 2>/dev/null
: > "$ERR_NFS" 2>/dev/null

(
    while true; do
        sleep 60
        # --append    transfers only bytes past the destination file size
        # --inplace   avoids rsync's rename-and-replace behavior (would
        #             break tail -F). Together: true in-place append.
        # timeout 30  guards against NFS stalls: if rsync blocks for >30s
        #             we kill it and retry next cycle.
        timeout 30 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null &
        timeout 30 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null &
        wait
    done
) &
MIRROR_PID=$!

cleanup() {
    kill "$MIRROR_PID" 2>/dev/null
    # Final flush on exit — longer timeout so the final log lands on NFS.
    timeout 60 rsync --append --inplace --quiet "$LOG_LOCAL" "$LOG_NFS" 2>/dev/null
    timeout 60 rsync --append --inplace --quiet "$ERR_LOCAL" "$ERR_NFS" 2>/dev/null
}
trap cleanup EXIT

echo "========================================"
echo "  RUN 9c: RESUME FROM ROUND 5"
echo "  Logs local: $LOG_LOCAL"
echo "  Mirrored to NFS every 60s (rsync --append): $LOG_NFS"
echo "  --learn-all-buffers (~448 params)"
echo "  Resuming from: $CKPT"
echo "  Started: $(date)"
echo "  Node: $(hostname)"
echo "  NFS nfsd threads: 64 (bumped from 8)"
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
echo "  RUN 9c COMPLETE — $(date)"
echo "========================================"
