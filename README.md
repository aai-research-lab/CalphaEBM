# CalphaEBM: Cα Energy-Based Model for Protein Dynamics

[![Tests](https://github.com/aai-research-lab/CalphaEBM/actions/workflows/tests.yml/badge.svg)](https://github.com/aai-research-lab/CalphaEBM/actions/workflows/tests.yml)
[![Documentation](https://github.com/aai-research-lab/CalphaEBM/actions/workflows/docs.yml/badge.svg)](https://aai-research-lab.github.io/CalphaEBM/)
[![PyPI version](https://badge.fury.io/py/calphaebm.svg)](https://badge.fury.io/py/calphaebm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

CalphaEBM is a **physics-grounded, modular energy function** for Cα protein coordinates that enables stable Langevin dynamics simulation. The model decomposes the effective free energy (potential of mean force) into four interpretable terms trained in phases.

## Key Features

- **Modular energy decomposition**: Local geometry, excluded volume, secondary structure, and packing
- **Phased training**: Freeze previous terms as new ones are added
- **Force-scale balancing**: Automatic λ recommendation from median force scales
- **PDB70-like dataset builder**: Create non-redundant training sets via RCSB API
- **Langevin dynamics**: Stable simulation with force clipping and diagnostics
- **Denoising score matching**: Train without partition function
- **Safety features**: Smooth switching, bounded radii, force caps
- **Multiple output formats**: DCD, NPY, PT, PDB for compatibility with MD tools

## Results on Ubiquitin (1UBQ)

After phased training (local → repulsion → secondary → packing):

| Metric | Value |
|--------|-------|
| No steric collapse | P(min < 3.8Å) = 0.0000 |
| Native contact retention | Q = 0.895 ± 0.039 |
| RMSD to native | 0.907 ± 0.311 Å |
| ΔRg (expansion) | 0.413 ± 0.163 Å |

## Installation

```bash
# Conda (recommended)
conda create -n calphaebm python=3.10
conda activate calphaebm
pip install calphaebm

# From source
git clone https://github.com/aai-research-lab/CalphaEBM.git
cd CalphaEBM
pip install -e ".[dev]"

## Quick Start

```
import calphaebm as cbm

# Load pre-trained model
model = cbm.load_checkpoint("checkpoints/run1/packing/step008000.pt")

# Simulate ubiquitin
traj = model.simulate(
    pdb="1ubq",
    steps=5000,
    step_size=2e-5,
    lambda_pack=0.1766
)

# Analyze
results = cbm.evaluate(traj, reference="snapshot_0000.xyz")
print(results.summary())
```
