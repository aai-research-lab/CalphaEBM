#!/bin/bash
#SBATCH --job-name=run3_sc
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/run3_sc_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536

INIT_CKPT=checkpoints/run2/run2/stage1_round002/step002000_best.pt
SC_DIR=checkpoints/run3_sc
N_ROUNDS=${1:-10}

echo "========================================"
echo "  SC RUN 3 — MALA + MULTI-β"
echo "  From: ${INIT_CKPT} (pre-SC, DSM+ELT only)"
echo "  Sampler: MALA (η=1e-4)"
echo "  Betas: [1, 10, 100, 1000] (from sc_defaults)"
echo "  Proteins: 64 (16/β), 100K steps"
echo "  Rg: native comparison, dead zone [0.7, 1.3]"
echo "  Frozen: removed"
echo "  ${N_ROUNDS} rounds"
echo "  Started: $(date)"
echo "========================================"

get_resume_round() {
    local marker="${SC_DIR}/run3/self-consistent/resume_next_round.txt"
    if [ -f "$marker" ]; then
        cat "$marker"
    else
        echo "0"
    fi
}

check_early_stop() {
    local consec_file="${SC_DIR}/run3/self-consistent/consecutive_increases.txt"
    if [ -f "$consec_file" ]; then
        local count=$(cat "$consec_file" | tr -d '[:space:]')
        if [ "$count" -ge 3 ] 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

for ROUND in $(seq 1 $N_ROUNDS); do
    if check_early_stop; then
        echo "[run3] Early stop fired — skipping round ${ROUND} and remaining"
        break
    fi

    RESUME_ROUND=$(get_resume_round)
    echo ""
    echo "[run3] Round ${ROUND}/${N_ROUNDS} (resume_round=${RESUME_ROUND}) — $(date)"
    echo ""

    calphaebm train \
        --stage sc \
        --resume "$INIT_CKPT" \
        --train-pdb train_hq.txt \
        --val-pdb val_hq.txt \
        --cache-dir pdb_cache \
        --processed-cache-dir processed_cache \
        --ckpt-dir $SC_DIR \
        --ckpt-prefix run3 \
        --energy-terms all \
        --local-window-size 8 \
        --elt-batch-size 32 \
        --n-rounds $ROUND \
        --collect-proteins 64 \
        --collect-steps 100000 \
        --collect-step-size 1e-4 \
        --collect-save-every 100 \
        --collect-max-len 512 \
        --n-workers 64 \
        --retrain-steps 2000 \
        --retrain-lr 1e-5 \
        --lambda-balance 0.01 \
        --lambda-discrim 2.0 \
        --lambda-sampled-hsm 1.0 \
        --lambda-sampled-qf 1.0 \
        --lambda-sampled-gap 2.0 \
        --lambda-native-depth 2.0 \
        --target-native-depth -3.0 \
        --lambda-rg 2.0 \
        --sc-eval-steps 5000 \
        --sc-eval-beta 100.0 \
        --sc-eval-proteins 64 \
        --sc-resume-round $RESUME_ROUND \
        --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
        --backbone-data-dir analysis/backbone_geometry/data \
        --secondary-data-dir analysis/secondary_analysis/data \
        --repulsion-data-dir analysis/repulsion_analysis/data \
        --packing-rg-lambda 10.0 \
        --coord-lambda 1.0

    echo "[run3] Round ${ROUND} done — $(date)"
done

echo "========================================"
echo "  SC RUN 3 COMPLETE — $(date)"
echo "========================================"
