#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 2: Repulsion calibration (controller-based, not gradient)
# ═══════════════════════════════════════════════════════════════════
#
# Calibrates λ_rep using a feedback controller that targets a specific
# fraction of C-alpha pairs inside the repulsive wall.
# Local term is frozen; only the repulsion gate/lambda is adjusted.
#
# Prereq: Phase 1 checkpoint
# Run:    bash scripts/train_repulsion_phase2.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run36}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Resume from Phase 1
RESUME="${CKPT_DIR}/run1/local/step005000.pt"

STEPS=2000
CKPT_EVERY=1000

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Phase 1 checkpoint not found: ${RESUME}"
    echo "Run Phase 1 first: bash scripts/train_local_phase1.sh"
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 2: Repulsion calibration"
echo "  Resume: ${RESUME}"
echo "  Steps: ${STEPS}"
echo "════════════════════════════════════════════════════════"

calphaebm train --phase repulsion \
    --resume "${RESUME}" \
    --pdb "${PDB_LIST}" \
    --steps "${STEPS}" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-every "${CKPT_EVERY}"

echo ""
echo "Phase 2 complete. Next: bash scripts/train_secondary_phase3.sh"
