#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 1: Local geometry training (θθ MLP + Δφ table + φφ MLP)
# ═══════════════════════════════════════════════════════════════════
#
# Architecture: 13,317 trainable params (3 λ scalars + 2 MLPs + embedding)
# Training:     DSM in IC space with multi-scale σ
# Subterms:     E_θθ (angular persistence), E_Δφ (torsional smoothness), E_φφ (consecutive torsion)
#
# Run:  bash scripts/train_local_phase1.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run36}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Training
STEPS=5000
LR=3e-4
LR_FINAL=3e-5
LR_SCHEDULE="cosine"

# Multi-scale IC noise (radians)
SIGMA_MIN=0.05
SIGMA_MAX=2.0

# Validation

# Checkpointing
CKPT_EVERY=2500

# Freeze non-local terms (safety — gates are also 0, but this ensures no grad flow)
FREEZE="repulsion secondary packing"

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 1: Local geometry training"
echo "  Run: ${RUN_NAME}  |  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
echo "  σ ∈ [${SIGMA_MIN}, ${SIGMA_MAX}] rad (multi-scale)"
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase local \
    --pdb "${PDB_LIST}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --lr-final "${LR_FINAL}" \
    --sigma-min-rad "${SIGMA_MIN}" \
    --sigma-max-rad "${SIGMA_MAX}" \
    --freeze ${FREEZE} \
    --validate-every 0 \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}"

echo ""
echo "Phase 1 complete. Next: bash scripts/train_repulsion_phase2.sh"
