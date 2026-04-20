#!/bin/bash
#SBATCH --job-name=run7b_sc
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run7b_sc_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

# ============================================================
# RUN 7b: RESUME SC TRAINING FROM ROUND 1
#
# Resumes from checkpoints/run7/run7/self-consistent/sc_round001.pt
# Same settings as run7_sc.sh — only difference is starting from
# round 2 with resume_round=1.
# ============================================================

SC_DIR=checkpoints/run7
N_ROUNDS=${1:-50}
START_ROUND=2

# Verify round 1 checkpoint exists
CKPT="${SC_DIR}/run7/self-consistent/sc_round001.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Round 1 checkpoint not found: $CKPT"
    exit 1
fi
echo "========================================"
echo "  RUN 7b: SC TRAINING — RESUME FROM ROUND 1"
echo "  PackingEnergy v6 + dRMSD-funnel"
echo "  64 proteins x 10K steps | 48 workers"
echo "  Rounds ${START_ROUND}..${N_ROUNDS} x 500 retrain steps"
echo "  beta ~ L*LogU[0.5,2.0] | balance ramp 1e-6->0.01"
echo "  Resuming from: $CKPT"
echo "  Started: $(date)"
echo "========================================"

get_resume_round() {
    local marker="${SC_DIR}/run7/self-consistent/resume_next_round.txt"
    if [ -f "$marker" ]; then
        cat "$marker"
    else
        echo "1"
    fi
}

for ROUND in $(seq $START_ROUND $N_ROUNDS); do
    RESUME_ROUND=$(get_resume_round)
    echo ""
    echo "[run7b] Round ${ROUND}/${N_ROUNDS} (resume_round=${RESUME_ROUND}) — $(date)"
    echo ""

    calphaebm train \
        --stage sc \
        --train-pdb train_hq.txt \
        --val-pdb val_hq.txt \
        --cache-dir pdb_cache \
        --processed-cache-dir processed_cache \
        --ckpt-dir $SC_DIR \
        --ckpt-prefix run7 \
        --energy-terms all \
        --local-window-size 8 \
        --n-rounds $ROUND \
        --collect-proteins 64 \
        --collect-steps 10000 \
        --collect-save-every 100 \
        --collect-max-len 512 \
        --max-negatives-per-protein 32 \
        --n-workers 48 \
        --retrain-steps 500 \
        --retrain-lr 1e-5 \
        --lambda-balance 0.01 \
        --lambda-discrim 2.0 \
        --lambda-sampled-hsm 1.0 \
        --lambda-sampled-qf 1.0 \
        --lambda-sampled-gap 2.0 \
        --lambda-sampled-drmsd-funnel 1.0 \
        --lambda-native-depth 2.0 \
        --target-native-depth -3.0 \
        --sc-margin 1.0 \
        --funnel-m 5.0 \
        --funnel-alpha 5.0 \
        --gap-m 5.0 \
        --gap-alpha 5.0 \
        --sc-eval-steps 50000 \
        --sc-eval-beta 100.0 \
        --sc-eval-proteins 16 \
        --sc-resume-round $RESUME_ROUND \
        --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
        --backbone-data-dir analysis/backbone_geometry/data \
        --secondary-data-dir analysis/secondary_analysis/data \
        --repulsion-data-dir analysis/repulsion_analysis/data \
        --packing-rg-lambda 1.0 \
        --packing-rho-lambda 1.0 \
        --hp-penalty-lambda 1.0 \
        --rho-penalty-lambda 1.0

    echo "[run7b] Round ${ROUND} done — $(date)"
done

echo "========================================"
echo "  RUN 7b SC COMPLETE — $(date)"
echo "========================================"
