#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 5: Full joint fine-tuning — 6-LOSS (Run43)
# ═══════════════════════════════════════════════════════════════════
#
# 6 losses:
#   1. DSM          — correct gradient direction (irreplaceable)
#   2. Native Depth — deep basin at native (exp target-based)
#   3. Gap          — absolute <E_decoy> - E_native gap (replaces Z-score)
#   4. Balance      — prevent term domination (safety rail, tiny)
#   5. Funnel       — E decreases as Q increases (structure-energy coupling)
#   6. Discrim      — per-subterm discrimination maintenance
#
# Z-score is DISABLED: it incentivises shrinking all energy scales
# to reduce decoy variance, rather than deepening the native well.
# Gap loss rewards the absolute energy difference instead.
#
# Disabled losses: frustration, basin, pack-C, Z-score
#
# Run43 changes vs run42:
#   1. Z-score loss REPLACED by gap loss (absolute gap, no variance trick)
#   2. Resume from run42 calibrated checkpoint (all fixes from run42)
#
# Prereq: cp checkpoints/run42/run42_calibrated.pt \
#            checkpoints/run43/run43_calibrated.pt
# Run:    bash scripts/training/train_full_phase5.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run43}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Resume from run42 calibrated (all fixes: ram gate, hbond, packing gate)
RESUME="${CKPT_DIR}/${RUN_NAME}_calibrated.pt"

# Training
STEPS=10000
LR=3e-4
LR_FINAL=3e-5
LR_SCHEDULE="cosine"
SCALAR_LR_MULT=1.0

# Multi-scale IC noise
SIGMA_MIN=0.05
SIGMA_MAX=2.0

# ── LOSS 1: DSM (always on, no lambda) ──────────────────────────

# ── LOSS 2: Native Depth ────────────────────────────────────────
# loss = exp(clamp(E_native - target, max=5))
LAMBDA_NATIVE_DEPTH=0.5
TARGET_NATIVE_DEPTH=-1.0

# ── LOSS 3: Gap (REPLACES Z-score) ──────────────────────────────
# loss = exp(-gap / T) where gap = <E_decoy/L> - E_native/L
# At gap=0.0 (no well):   loss=exp(0)=1.0 → strong push
# At gap=0.5:             loss=exp(-1.0)=0.37 → good gradient
# At gap=1.0 (target):    loss=exp(-2.0)=0.14 → diminishing
# At gap=2.0:             loss=exp(-4.0)=0.02 → self-annealing
# Unlike Z-score, this rewards absolute gap, not gap/σ.
LAMBDA_GAP=1.0
GAP_T=0.5

# Z-score DISABLED (causes energy scale collapse)
LAMBDA_ZSCORE=0.0
TARGET_ZSCORE=3.0

# ── LOSS 4: Balance (safety rail) ───────────────────────────────
LAMBDA_BALANCE=0.001

# ── LOSS 5: Funnel (Q-energy coupling) ──────────────────────────
# Reduced from 1.0 to 0.5: funnel now operates at correct scale
# (double-norm fix), so full weight overwhelms pack_geom discrim.
LAMBDA_FUNNEL=0.5
FUNNEL_T=0.5
ELT_EVERY=2
ELT_MAX_LEN=512
ELT_BATCH_SIZE=16

# ── LOSS 6: Per-subterm discrimination maintenance ─────────────
# L = max_k softplus(E_k(native) - E_k(perturbed))
# Prevents individual subterms from collapsing during joint training.
# Self-annealing: zero gradient for terms already discriminating.
# Increased from 1.0 to 2.0: funnel now operates at correct scale,
# needs stronger counterbalance to prevent pack_geom collapse.
# Changed from mean to max: focuses gradient on the worst subterm.
LAMBDA_DISCRIM=2.0
DISCRIM_EVERY=2

# ── DISABLED LOSSES ─────────────────────────────────────────────
LAMBDA_FRUST=0.0
LAMBDA_BASIN=0.0
LAMBDA_PACK_C=0.0

# ── DISABLED SUBTERMS ─────────────────────────────────────────────
# 4-mer architecture: no separate theta_theta or delta_phi to disable
DISABLE_SUBTERMS=""

# ── Lambda floor ────────────────────────────────────────────────
LAMBDA_HB_BETA_FLOOR=0.1

# ── Validation ──────────────────────────────────────────────────
VAL_EVERY=2500
VAL_SAMPLES=8
LANGEVIN_BETA=10.0
LANGEVIN_STEPS=500

# ── Checkpointing ──────────────────────────────────────────────
CKPT_EVERY=2000

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Checkpoint not found: ${RESUME}"
    echo ""
    echo "To start from run42 calibrated:"
    echo "  mkdir -p checkpoints/run43"
    echo "  cp checkpoints/run42/run42_calibrated.pt checkpoints/run43/run43_calibrated.pt"
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 5: 6-Loss Training (Run43)"
echo "  Resume: ${RESUME}"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
echo ""
echo "  Active losses:"
echo "    1. DSM          (always on)"
echo "    2. Native Depth λ=${LAMBDA_NATIVE_DEPTH} target=${TARGET_NATIVE_DEPTH}"
echo "    3. Gap          λ=${LAMBDA_GAP} T=${GAP_T}"
echo "    4. Balance      λ=${LAMBDA_BALANCE}"
echo "    5. Funnel       λ=${LAMBDA_FUNNEL} T=${FUNNEL_T} (reduced — correct scale now)"
echo "    6. Discrim      λ=${LAMBDA_DISCRIM} every ${DISCRIM_EVERY} steps (max mode)"
echo ""
echo "  Disabled losses: frustration, basin, pack-C, Z-score"
echo "  Disabled subterms: ${DISABLE_SUBTERMS}"
echo "  hb_beta floor: ${LAMBDA_HB_BETA_FLOOR}"
echo "  Validation: every ${VAL_EVERY} steps, β=${LANGEVIN_BETA}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase full \
    --resume "${RESUME}" --resume-model-only \
    --pdb "${PDB_LIST}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --lr-final "${LR_FINAL}" \
    --scalar-lr-mult "${SCALAR_LR_MULT}" \
    --sigma-min-rad "${SIGMA_MIN}" \
    --sigma-max-rad "${SIGMA_MAX}" \
    --elt-max-len "${ELT_MAX_LEN}" \
    --elt-batch-size "${ELT_BATCH_SIZE}" \
    --elt-every "${ELT_EVERY}" \
    --lambda-funnel "${LAMBDA_FUNNEL}" \
    --funnel-T "${FUNNEL_T}" \
    --lambda-zscore "${LAMBDA_ZSCORE}" \
    --target-zscore "${TARGET_ZSCORE}" \
    --lambda-gap "${LAMBDA_GAP}" \
    --gap-T "${GAP_T}" \
    --lambda-frustration "${LAMBDA_FRUST}" \
    --lambda-native-depth "${LAMBDA_NATIVE_DEPTH}" \
    --target-native-depth "${TARGET_NATIVE_DEPTH}" \
    --lambda-balance "${LAMBDA_BALANCE}" \
    --lambda-discrim "${LAMBDA_DISCRIM}" \
    --discrim-every "${DISCRIM_EVERY}" \
    --discrim-mode max \
    --lambda-basin "${LAMBDA_BASIN}" \
    --lambda-pack-contrastive "${LAMBDA_PACK_C}" \
    --lambda-hb-beta-floor "${LAMBDA_HB_BETA_FLOOR}" \
    ${DISABLE_SUBTERMS:+--disable-subterms ${DISABLE_SUBTERMS}} \
    --langevin-beta "${LANGEVIN_BETA}" \
    --val-langevin-steps "${LANGEVIN_STEPS}" \
    --val-max-samples "${VAL_SAMPLES}" \
    --validate-every "${VAL_EVERY}" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}"

echo ""
echo "Phase 5 complete."
echo "Run43 targets: RMSD < 4Å @ β=10, gap > 1.0 E/res, E_native < -1.0/res"
