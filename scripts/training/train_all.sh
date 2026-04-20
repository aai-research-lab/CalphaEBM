#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# CαEBM: Complete phased training pipeline
# ═══════════════════════════════════════════════════════════════════
#
# Architecture: 7,362 params, 7 subterms, 4-mer θφ MLP
#   local(7,362p):   E_θφ 4-mer MLP + Ramachandran validity gate
#   repulsion(1p):   tabulated wall
#   secondary(589p): E_ram(4 basins) + E_hb_α(p²,σ≥0.05) + E_hb_β
#   packing(1,625p): E_geom MLP + E_contact(SVD) + Ramachandran gate
#
# Phases:
#   1.  Local       → learn backbone geometry via DSM
#   2.  Repulsion   → calibrate excluded volume (measure-set-verify)
#   3.  Secondary   → learn Ramachandran basins + H-bond coupling
#   4.  Packing     → contrastive pre-training of geom MLP + contacts
#   4b. Calibration → set all λ so each subterm ≈ 1/7 of total E/res
#   5.  Full+ELT    → joint fine-tuning with 5 losses + discrim
#
# Usage:
#   export RUN_NAME=run42
#   bash scripts/training/train_all.sh           # run all phases
#   bash scripts/training/train_all.sh 3         # start from phase 3
#   bash scripts/training/train_all.sh 4b        # start from calibration
#   bash scripts/training/train_all.sh 5         # start from phase 5
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START="${1:-1}"
RUN_NAME="${RUN_NAME:?ERROR: Set RUN_NAME before running (e.g. export RUN_NAME=run42)}"

echo "═══════════════════════════════════════════════════════════"
echo "  CαEBM ${RUN_NAME}: Phased Training Pipeline"
echo "  Starting from phase: ${START}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Helper: compare phase ordering (1 < 2 < 3 < 4 < 4b < 5)
should_run() {
    local phase="$1"
    case "${START}" in
        1)                          return 0 ;;
        2)   [[ "$phase" != "1" ]]  && return 0 || return 1 ;;
        3)   [[ "$phase" =~ ^[3-5]$|^4b$ ]] && return 0 || return 1 ;;
        4)   [[ "$phase" =~ ^[4-5]$|^4b$ ]] && return 0 || return 1 ;;
        4b)  [[ "$phase" == "4b" || "$phase" == "5" ]] && return 0 || return 1 ;;
        5)   [[ "$phase" == "5" ]]  && return 0 || return 1 ;;
        *)   return 1 ;;
    esac
}

phase_start() {
    echo ""
    echo "▶━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ Phase $1: $2"
    echo "▶━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

if should_run 1; then
    phase_start "1" "Local geometry (DSM, 4-mer θφ MLP)"
    bash "${SCRIPT_DIR}/train_local_phase1.sh"
fi

if should_run 2; then
    phase_start "2" "Repulsion calibration (measure-set-verify)"
    bash "${SCRIPT_DIR}/train_repulsion_phase2.sh"
fi

if should_run 3; then
    phase_start "3" "Secondary structure (ram + hb_α + hb_β)"
    bash "${SCRIPT_DIR}/train_secondary_phase3.sh"
fi

if should_run 4; then
    phase_start "4" "Packing contrastive (geom MLP + contacts)"
    bash "${SCRIPT_DIR}/train_packing_phase4.sh"
fi

if should_run 4b; then
    phase_start "4b" "Calibration (target 1/7 per subterm)"
    bash "${SCRIPT_DIR}/calibrate_phase4b.sh"
fi

if should_run 5; then
    phase_start "5" "Full 5-loss + discrim (DSM+depth+Z+funnel+balance+discrim)"
    bash "${SCRIPT_DIR}/train_full_phase5.sh"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ${RUN_NAME}: All phases complete."
echo "═══════════════════════════════════════════════════════════"
