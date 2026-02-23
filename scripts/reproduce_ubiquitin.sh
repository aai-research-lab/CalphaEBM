#!/bin/bash
# Reproduce ubiquitin results from V4 report

set -e  # Exit on error

echo "Reproducing ubiquitin trajectory from CalphaEBM V4 report"
echo "=========================================================="

# Check if checkpoint exists
CKPT="checkpoints_energy/run1/packing/step008000.pt"
if [ ! -f "$CKPT" ]; then
    echo "Error: Checkpoint not found at $CKPT"
    echo "Please train model first or download pre-trained checkpoint"
    exit 1
fi

# Create output directory
OUT_DIR="runs/ubq_pack0p1766_stepsize2e-5"
mkdir -p $OUT_DIR

# Run simulation
echo "Running simulation..."
calphaebm simulate \
    --pdb 1ubq \
    --ckpt $CKPT \
    --out-dir $OUT_DIR \
    --steps 5000 \
    --step-size 2e-5 \
    --lambda-local 1.0 \
    --lambda-rep 1.0 \
    --lambda-ss 2.789315 \
    --lambda-pack 0.176610 \
    --repulsion-mode learned-radius \
    --log-every 50 \
    --save-every 50

echo "Simulation complete!"

# Evaluate trajectory
echo "Evaluating trajectory..."
calphaebm evaluate \
    --traj $OUT_DIR \
    --ref-xyz $OUT_DIR/snapshot_0000.xyz \
    --contact-cutoff 8.0 \
    --exclude 2 \
    --rdf-rmax 20 \
    --rdf-dr 0.25 \
    --burnin 10

echo "Evaluation complete! Results saved to $OUT_DIR/eval/"
