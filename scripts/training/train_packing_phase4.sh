#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 4: Packing contrastive pre-training (E_geom MLP + E_contact)
# ═══════════════════════════════════════════════════════════════════
#
# Trains: E_geom (geometry MLP, 1601p) + E_contact (rank-1 pair, 23p, SVD init)
# Uses contrastive loss: native vs sequence-shuffle and IC-noise negatives.
# Local, repulsion, secondary are frozen.
#
# Prereq: Phase 3 checkpoint
# Run:    bash scripts/train_packing_phase4.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run36}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Resume from Phase 3
RESUME="${CKPT_DIR}/run1/secondary/step005000.pt"

# Training
STEPS=5000
LR=3e-4
LR_FINAL=3e-5
LR_SCHEDULE="cosine"

# Multi-scale IC noise
SIGMA_MIN=0.05
SIGMA_MAX=2.0

# Packing contrastive loss
LAMBDA_PACK_C=1.0
PACK_C_T=2.0

# Validation

CKPT_EVERY=2500

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Phase 3 checkpoint not found: ${RESUME}"
    echo "Run Phase 3 first: bash scripts/train_secondary_phase3.sh"
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 4: Packing contrastive pre-training"
echo "  Resume: ${RESUME}"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
echo "  Contrastive: λ=${LAMBDA_PACK_C}, T=${PACK_C_T}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase packing \
    --resume "${RESUME}" \
    --pdb "${PDB_LIST}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --lr-final "${LR_FINAL}" \
    --sigma-min-rad "${SIGMA_MIN}" \
    --sigma-max-rad "${SIGMA_MAX}" \
    --lambda-pack-contrastive "${LAMBDA_PACK_C}" \
    --pack-contrastive-T-base "${PACK_C_T}" \
    --freeze local repulsion secondary \
    --validate-every 0 \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}"

echo ""
echo "Phase 4 complete. Next: bash scripts/train_full_phase5.sh"
