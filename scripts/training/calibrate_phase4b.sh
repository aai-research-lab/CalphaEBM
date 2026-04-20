#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Phase 4b: Calibration (between packing and full joint training)
# ═══════════════════════════════════════════════════════════════════
#
# Measures E/res for each of 9 subterms on native structures using
# the trained Phase 4 checkpoint, then sets λ so each produces
# target = 1/9 ≈ 0.111 E/res.
#
# Key behaviours:
#   - Uses TRAINED checkpoint (not fresh model) for measurement
#   - Skips θθ/φφ MLPs (near-zero at init, can't calibrate)
#   - Preserves λ_rep from Phase 2 (safety constraint, not recalibrated)
#   - Caps all λ at 1.5 (Hessian safety)
#
# Prereq: Phase 4 checkpoint
# Run:    bash scripts/training/calibrate_phase4b.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-run36}"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"

# Input: Phase 4 trained checkpoint
INPUT_CKPT="${CKPT_DIR}/run1/packing/step005000.pt"

# Output: calibrated checkpoint for Phase 5
OUTPUT_CKPT="${CKPT_DIR}/${RUN_NAME}_calibrated.pt"
OUTPUT_JSON="calibration_${RUN_NAME}.json"

# ── Verify prerequisite ─────────────────────────────────────────
if [ ! -f "${INPUT_CKPT}" ]; then
    echo "ERROR: Phase 4 checkpoint not found: ${INPUT_CKPT}"
    echo "Run Phase 4 first: bash scripts/training/train_packing_phase4.sh"
    exit 1
fi

# ── Launch calibration ───────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Phase 4b: Calibration"
echo "  Input:  ${INPUT_CKPT}"
echo "  Output: ${OUTPUT_CKPT}"
echo "  Target: 1/7 ≈ 0.143 E/res per subterm (7 subterms with 4-mer)"
echo "  Cap:    1.5 (Hessian safety)"
echo "  Rep:    preserved from Phase 2"
echo "  Skip:   none (4-mer architecture, all subterms calibrated)"
echo "════════════════════════════════════════════════════════"

calphaebm calibrate \
    --pdb "${PDB_LIST}" \
    --apply-to-ckpt "${INPUT_CKPT}" \
    --out-ckpt "${OUTPUT_CKPT}" \
    --out "${OUTPUT_JSON}" \
    --target 0.142857

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Calibration complete."
echo "  Checkpoint: ${OUTPUT_CKPT}"
echo "  Summary:    ${OUTPUT_JSON}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Next: bash scripts/training/train_full_phase5.sh"
