#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 3: Secondary structure training (Ramachandran + H-bonds)
# ═══════════════════════════════════════════════════════════════════
#
# Trains: E_ram (mixture-of-basins), E_hb_α (helix H-bond), E_hb_β (sheet H-bond)
# Local and repulsion are frozen; only secondary parameters train.
#
# Prereq: Phase 2 checkpoint
# Run:    bash scripts/training/train_secondary_phase3.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run36}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Resume from Phase 2 (single calibration step)
RESUME="${CKPT_DIR}/run1/repulsion/step000001.pt"

# Training
STEPS=5000
LR=3e-4
LR_FINAL=3e-5
LR_SCHEDULE="cosine"

# Multi-scale IC noise (must match Phase 1 / Phase 5)
SIGMA_MIN=0.05
SIGMA_MAX=2.0

# Checkpointing
CKPT_EVERY=2500

# Freeze non-secondary terms
FREEZE="local repulsion packing"

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Phase 2 checkpoint not found: ${RESUME}"
    echo "Run Phase 2 first: bash scripts/training/train_repulsion_phase2.sh"
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 3: Secondary structure training"
echo "  Resume: ${RESUME}"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
echo "  σ ∈ [${SIGMA_MIN}, ${SIGMA_MAX}] rad (multi-scale)"
echo "  Freeze: ${FREEZE}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase secondary \
    --resume "${RESUME}" \
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
echo "Phase 3 complete. Next: bash scripts/training/train_packing_phase4.sh"
