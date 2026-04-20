#!/bin/bash
#SBATCH --job-name=run6_r4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/run6_full_stage_%j.log

source ~/calphaebm_env/bin/activate
export LD_LIBRARY_PATH=$(find ~/calphaebm_env/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
cd ~/CalphaEBM
ulimit -n 65536
export CALPHAEBM_WORKERS=16  # 16 CPUs for decoy generation

# ============================================================
# RUN6: Full-stage training (clean restart)
# ============================================================
# Changes from Run5:
#   - PackingEnergy v6: 5-group physicochemical scheme
#       Group 0 core_hydrophobic    [PHE,ILE,LEU,MET,VAL]
#       Group 1 amphipathic_hp      [ALA,PRO,TRP,TYR]
#       Group 2 positive            [HIS,LYS,ARG]
#       Group 3 negative            [ASP,GLU]
#       Group 4 polar               [CYS,GLY,ASN,GLN,SER,THR]
#     E_hp: product Gaussian over n_i^(k) per group (22 params)
#     E_rho: product Gaussian over rho^(k) per group (1 param)
#     E_hp_pen: per-residue per-group exponential penalty
#     E_rho_pen: per-chain per-group exponential penalty
#     coord_n_star.json recomputed with 5-group statistics
#   - dRMSD-funnel replaces Rg-funnel (--lambda-drmsd):
#       Full pairwise dRMSD — topology-sensitive
#       Cannot be satisfied by compact misfolds with wrong topology
#       Free computation — reuses distance matrix from Q
#       threshold=0.5 Å (was 0.05 dimensionless for Rg)
#   - steps-per-round 5000 (was 2000 in Run5, 3000 initial Run6)
# ============================================================

echo "========================================"
echo "  RUN 6: FULL-STAGE TRAINING (RESUME FROM ROUND 3)"
echo "  PackingEnergy v6 + 5-group product Gaussian"
echo "  dRMSD-funnel replaces Rg-funnel"
echo "  Saturating exponential margins (m=5, α=5)"
echo "  Sampler: MALA"
echo "  Resumed from: stage1_round003/step005000.pt
  lambda-qf/drmsd: 1.0 → 0.1 (funnel dominance fix)
  Started: $(date)"
echo "========================================"

calphaebm train \
    --stage full \
    --train-pdb train_hq.txt \
    --val-pdb val_hq.txt \
    --cache-dir pdb_cache \
    --processed-cache-dir processed_cache \
    --ckpt-dir checkpoints/run6 \
    --ckpt-prefix run6 \
    --energy-terms all \
    --local-window-size 8 \
    --lr 5e-4 \
    --lr-final 5e-5 \
    --lr-schedule cosine \
    --max-rounds 5 \
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
    --packing-rg-lambda 1.0 \
    --packing-rho-lambda 1.0 \
    --hp-penalty-lambda 1.0 \
    --rho-penalty-lambda 1.0 \
    --collect-proteins 1024 \
    --elt-max-len 512 \
    --collect-max-len 512 \
    --resume checkpoints/run6/run6/stage1_round003/step005000.pt \
    --resume-model-only

echo "========================================"
echo "  RUN 6 FULL-STAGE COMPLETE — $(date)"
echo "========================================"
