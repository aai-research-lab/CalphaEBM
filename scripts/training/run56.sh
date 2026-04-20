#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  Run56: Fresh Training — No Contact Energy
#
#  FROM SCRATCH — no resume. E_geom learns packing preferences
#  from DSM + ELT losses alone, without E_contact ever corrupting
#  the gradient signal.
#
#  Why: Run55 showed that disabling E_contact after run52 training
#  only marginally helped (83% vs 80% Rg at 100K) because E_geom's
#  1601 params inherited compaction bias from run52 where E_contact
#  was active. The MLP learned "more neighbors = better" alongside
#  the monotonic sigmoid. Must train from scratch so E_geom never
#  sees the compaction signal.
#
#  Architecture: 7 learned subterms → 6 (no contact)
#    E = E_local(θφ) + E_secondary(ram+hb_α+hb_β) + E_repulsion
#      + E_packing(geom) + E_constraint(coord+Rg)
#
#  Training: gentle α-DSM [0.85, 1.10] + full ELT losses
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════"
echo "  Run56: Fresh Training — No Contact Energy"
echo "  FROM SCRATCH (no resume)"
echo "  DISABLED: E_contact (never active)"
echo "  α ~ U(0.85, 1.10) — gentle 3× DSM"
echo "  Steps: 20000  |  LR: 6e-4 → 6e-5 cosine"
echo "════════════════════════════════════════════════════════"

mkdir -p logs

calphaebm train --phase full \
    --pdb train_entities_9600.txt \
    --steps 20000 \
    --lr 6e-4 --lr-schedule cosine --lr-final 6e-5 \
    --scalar-lr-mult 1.0 \
    \
    `# ── DSM noise + gentle α augmentation ──` \
    --sigma-min-rad 0.05 --sigma-max-rad 2.0 \
    --dsm-alpha-min 0.85 --dsm-alpha-max 1.10 \
    \
    `# ── Dataset filters ──` \
    --max-rg-ratio 1.3 \
    --elt-max-len 512 --elt-batch-size 16 --elt-every 2 \
    \
    `# ── Packing: E_geom + E_Rg + E_coord (NO E_contact) ──` \
    --packing-rg-lambda 0.1 \
    --packing-rg-r0 2.0 \
    --packing-rg-nu 0.38 \
    --coord-lambda 0.01 \
    --coord-n-star-file analysis/coordination_analysis/coord_n_star.json \
    --lambda-rg 0.0 \
    --disable-subterms contact \
    \
    `# ── ELT losses ──` \
    --lambda-funnel 0.5 --funnel-T 0.5 \
    --lambda-zscore 0.0 --target-zscore 3.0 \
    --lambda-gap 1.0 --gap-T 0.5 \
    --lambda-frustration 0.0 \
    --lambda-native-depth 0.5 --target-native-depth -1.0 \
    \
    `# ── Per-subterm discrimination ──` \
    --lambda-discrim 2.0 --discrim-every 2 --discrim-mode max \
    \
    `# ── Balance (6 learned subterms, r=7.0) ──` \
    --lambda-balance 0.001 --balance-r 7.0 \
    \
    `# ── Lambda floors ──` \
    --lambda-hb-beta-floor 0.1 \
    \
    `# ── Disabled losses ──` \
    --lambda-basin 0.0 \
    --lambda-pack-contrastive 0.0 \
    \
    `# ── Validation ──` \
    --langevin-beta 50.0 \
    --val-langevin-steps 500 \
    --val-max-samples 8 \
    --validate-every 500 \
    \
    `# ── Checkpoints ──` \
    --ckpt-dir checkpoints/run56 \
    --ckpt-every 500 \
    \
    2>&1 | tee logs/run56.log
