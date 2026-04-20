#!/bin/bash
#SBATCH --job-name=run8_sc
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run8_sc_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=48

# ============================================================
# RUN 8: SELF-CONSISTENT TRAINING FROM SCRATCH
#
# Key change from Run7: β = L × LogU[0.01, 1.0] (was [0.5, 2.0])
#   Low β → rapid unfolding → negatives at Q=0.2–0.5
#   High β → near-native → negatives at Q=0.85–0.97
#   Full deformation spectrum for local MLP discrimination
#
# Collection:  64 proteins × 10K MALA steps per round
#              β = L × LogU[0.01, 1.0] (set in defaults.py)
#              save every 100 steps → 100 snapshots/protein
#              max 32 failures per protein
#              48 workers | sampler=mala
#
# Retraining:  500 steps/round
#              50 rounds → 25K gradient steps
#              lambda_balance ramped 1e-6 → 0.01 over 10 rounds
#
# Losses:
#   PDB:        depth=2.0 | discrim=2.0
#   SC sampled: gap=2.0 | drmsd_funnel=1.0 | hsm=1.0 | qf=1.0
#   Margins:    funnel(m=5,a=5) | gap(m=5,a=5) | sc_margin=1.0
#
# Checkpoints: checkpoints/run8/run8/self-consistent/sc_round{N:03d}.pt
# Eval decoupled — run8_eval_watcher.sh as separate SLURM job.
# ============================================================

SC_DIR=checkpoints/run8
N_ROUNDS=${1:-50}

echo "========================================"
echo "  RUN 8: SC TRAINING FROM SCRATCH"
echo "  Wide β range: L × LogU[0.01, 1.0]"
echo "  PackingEnergy v6 + dRMSD-funnel"
echo "  64 proteins × 10K steps | 48 workers"
echo "  ${N_ROUNDS} rounds × 500 retrain steps"
echo "  Training from scratch — no resume"
echo "  Eval decoupled (run8_eval_watcher.sh)"
echo "  Started: $(date)"
echo "========================================"

get_resume_round() {
    local marker="${SC_DIR}/run8/self-consistent/resume_next_round.txt"
    if [ -f "$marker" ]; then
        cat "$marker"
    else
        echo "0"
    fi
}

for ROUND in $(seq 1 $N_ROUNDS); do
    RESUME_ROUND=$(get_resume_round)
    echo ""
    echo "[run8] Round ${ROUND}/${N_ROUNDS} (resume_round=${RESUME_ROUND}) — $(date)"
    echo ""

    calphaebm train \
        --stage sc \
        --train-pdb train_hq.txt \
        --val-pdb val_hq.txt \
        --cache-dir pdb_cache \
        --processed-cache-dir processed_cache \
        --ckpt-dir $SC_DIR \
        --ckpt-prefix run8 \
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

    echo "[run8] Round ${ROUND} done — $(date)"
done

echo "========================================"
echo "  RUN 8 SC COMPLETE — $(date)"
echo "========================================"
