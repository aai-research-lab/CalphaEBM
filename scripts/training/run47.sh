#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run47: Rg gate — first training run with packing Rg suppression
# ═══════════════════════════════════════════════════════════════════
#
# Run46 achieved (final EMA):
#   Z̄=1.91  gap=0.799  |slope|=2.12  af%=4%
#   Instant Z crossed 3.0 twice (s7000: Z=2.96, s9600: Z=3.03)
#   Best composite: s9500 (1.094) — dphi=0.910, Q=0.992, RMSD=1.95
#   Energy balance: pack=38% ss=33% local=25% rep=4%
#
# Problem: 100K basin tests show compaction (Rg 9.6→6.5 at β=100).
# 2K dynamics look great but long-timescale packing overcorrects.
#
# Solution: Rg Gaussian gate on packing terms (--rg-gate-sigma 0.10)
#   gate = exp(-(Rg/Rg_native - 1)^2 / (2 * 0.10^2))
#   Calibrated from 500 structures: σ_Rg=0.058 at σ_noise=0.05 rad
#   Using 0.10 (conservative) — 88% active at ±5%, 14% at ±20%
#
# No new parameters. Gate computed from coordinates each forward pass.
# Old checkpoint loads cleanly — rg_gate_sigma is a constructor float.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

RUN_NAME="run47"
PDB_LIST="train_entities.no_test_entries.txt"
CKPT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" logs

RESUME="checkpoints/run46/run1/full/step009500.pt"

STEPS=10000
LR=6e-4
LR_FINAL=6e-5
LR_SCHEDULE="cosine"
SCALAR_LR_MULT=1.0

SIGMA_MIN=0.05
SIGMA_MAX=2.0

# ── Rg gate (NEW in run47) ────────────────────────────────────────
RG_GATE_SIGMA=0.10

LAMBDA_NATIVE_DEPTH=0.5
TARGET_NATIVE_DEPTH=-1.0

LAMBDA_GAP=1.0
GAP_T=0.5

LAMBDA_ZSCORE=0.0
TARGET_ZSCORE=3.0

LAMBDA_BALANCE=0.001

LAMBDA_FUNNEL=0.5
FUNNEL_T=0.5
ELT_EVERY=2
ELT_MAX_LEN=512
ELT_BATCH_SIZE=16

LAMBDA_DISCRIM=2.0
DISCRIM_EVERY=2

LAMBDA_FRUST=0.0
LAMBDA_BASIN=0.0
LAMBDA_PACK_C=0.0

LAMBDA_HB_BETA_FLOOR=0.1

VAL_EVERY=500
VAL_SAMPLES=8
LANGEVIN_BETA=50.0
LANGEVIN_STEPS=500

CKPT_EVERY=500

if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Checkpoint not found: ${RESUME}"
    exit 1
fi

echo "════════════════════════════════════════════════════════"
echo "  Run47: Rg gate on packing (σ=${RG_GATE_SIGMA})"
echo "  Resume: ${RESUME} (model only, fresh optimizer)"
echo "  Steps: ${STEPS}  |  LR: ${LR}→${LR_FINAL}"
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
    --rg-gate-sigma "${RG_GATE_SIGMA}" \
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
    2>&1 | tee logs/run47.log

echo ""
echo "Run47 complete."
