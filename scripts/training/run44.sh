#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run44: Continue from run43 s7500 with 2× LR
# ═══════════════════════════════════════════════════════════════════
#
# Goal: Let packing rebalance. β sweep showed systematic Rg compaction
# (9.2→7.4 Å). Packing was declining (53%→46%) during run43 but the
# learning rate died before it could finish. 2× LR gives budget to
# complete the rebalancing.
#
# Changes from run43:
#   - LR: 3e-4 → 6e-4 (2×)
#   - LR final: 3e-5 → 6e-5 (2×)
#   - Steps: 10000 → 5000 (shorter, monitor closely)
#   - Validate: every 500 (was 2500) — DynamicsValidator at β=100
#   - Checkpoint: every 500 (was 2000)
#   - GenerationValidator: β=50 (was β=10), 500 steps
#   - Fresh optimizer (--resume-model-only)
#
# Same loss weights as run43 (EXACT):
#   L = L_DSM + 0.5·L_funnel + 1.0·L_gap + 0.5·L_depth
#       + 0.001·L_balance + 2.0·L_discrim(max)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="run44"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" logs

# Resume from run43 s7500 (best landscape checkpoint)
RESUME="checkpoints/run44/run43_step007500.pt"

# Training — 2× LR, shorter run, frequent checkpoints
STEPS=5000
LR=6e-4
LR_FINAL=6e-5
LR_SCHEDULE="cosine"
SCALAR_LR_MULT=1.0       # same as run43 — base LR already 2×

# Multi-scale IC noise (same as run43)
SIGMA_MIN=0.05
SIGMA_MAX=2.0

# ── LOSS 1: DSM (always on) ─────────────────────────────────────

# ── LOSS 2: Native Depth (same as run43) ────────────────────────
LAMBDA_NATIVE_DEPTH=0.5
TARGET_NATIVE_DEPTH=-1.0

# ── LOSS 3: Gap (same as run43) ─────────────────────────────────
LAMBDA_GAP=1.0
GAP_T=0.5

# Z-score DISABLED (same as run43)
LAMBDA_ZSCORE=0.0
TARGET_ZSCORE=3.0

# ── LOSS 4: Balance (same as run43) ─────────────────────────────
LAMBDA_BALANCE=0.001

# ── LOSS 5: Funnel (same as run43) ──────────────────────────────
LAMBDA_FUNNEL=0.5
FUNNEL_T=0.5
ELT_EVERY=2
ELT_MAX_LEN=512
ELT_BATCH_SIZE=16

# ── LOSS 6: Discrim (same as run43) ─────────────────────────────
LAMBDA_DISCRIM=2.0
DISCRIM_EVERY=2

# ── DISABLED LOSSES (same as run43) ─────────────────────────────
LAMBDA_FRUST=0.0
LAMBDA_BASIN=0.0
LAMBDA_PACK_C=0.0

# ── Lambda floor (same as run43) ────────────────────────────────
LAMBDA_HB_BETA_FLOOR=0.1

# ── Validation — CHANGED ────────────────────────────────────────
# More frequent: every 500 steps (was 2500)
# DynamicsValidator: β=100 on crambin (NEW, runs automatically)
# GenerationValidator: β=50 (was β=10), 500 steps
VAL_EVERY=500
VAL_SAMPLES=8
LANGEVIN_BETA=50.0
LANGEVIN_STEPS=500

# ── Checkpointing — CHANGED ────────────────────────────────────
CKPT_EVERY=500

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Checkpoint not found: ${RESUME}"
    echo ""
    echo "Create it from run43 s7500:"
    echo "  mkdir -p checkpoints/run44"
    echo "  cp checkpoints/run43/<path-to-step007500.pt> checkpoints/run44/run44_calibrated.pt"
    echo ""
    echo "Or point RESUME to the actual path, e.g.:"
    echo "  RESUME=checkpoints/run42/run1/full/step007500.pt bash scripts/training/train_run44.sh"
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Run44: 2× LR rebalancing from run43 s7500"
echo "  Resume: ${RESUME} (model only, fresh optimizer)"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL} (2× run43)"
echo ""
echo "  Active losses (SAME as run43):"
echo "    1. DSM          (always on)"
echo "    2. Native Depth λ=${LAMBDA_NATIVE_DEPTH} target=${TARGET_NATIVE_DEPTH}"
echo "    3. Gap          λ=${LAMBDA_GAP} T=${GAP_T}"
echo "    4. Balance      λ=${LAMBDA_BALANCE}"
echo "    5. Funnel       λ=${LAMBDA_FUNNEL} T=${FUNNEL_T}"
echo "    6. Discrim      λ=${LAMBDA_DISCRIM} every ${DISCRIM_EVERY} steps (max mode)"
echo ""
echo "  CHANGED: LR 2×, validate every ${VAL_EVERY}, β=${LANGEVIN_BETA}"
echo "  NEW: DynamicsValidator at β=100 on crambin (automatic)"
echo "  Watch: packing% ↓, Rg ratio ↑, RMSF stable, E_delta < 0"
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
    --langevin-beta "${LANGEVIN_BETA}" \
    --val-langevin-steps "${LANGEVIN_STEPS}" \
    --val-max-samples "${VAL_SAMPLES}" \
    --validate-every "${VAL_EVERY}" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}" \
    2>&1 | tee logs/run44.log

echo ""
echo "Run44 complete."
echo "Check: dynamics_frac_packing declining, dynamics_rg_ratio increasing"
