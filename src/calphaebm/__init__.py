"""CalphaEBM: Cα Energy-Based Model for Protein Backbone Dynamics.

This package provides a modular, physics-grounded energy function for
coarse-grained protein simulations using Cα coordinates only.
"""

from calphaebm._version import __version__

# Data utilities
from calphaebm.data.synthetic import make_extended_chain, random_sequence
from calphaebm.evaluation.metrics.contacts import contact_count, q_hard, q_smooth
from calphaebm.evaluation.metrics.rg import radius_of_gyration

# Evaluation
from calphaebm.evaluation.metrics.rmsd import rmsd_kabsch
from calphaebm.evaluation.reporting import EvaluationReport, TrajectoryEvaluator
from calphaebm.geometry.dihedral import dihedral

# Geometry (commonly used functions)
from calphaebm.geometry.internal import bond_angles, bond_lengths, torsions
from calphaebm.models.cross_terms import SecondaryStructureEnergy

# Models
from calphaebm.models.energy import TotalEnergy
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import (
    RepulsionEnergy,
    RepulsionEnergyFixed,
    RepulsionEnergyLearnedRadius,
)

# Simulation
from calphaebm.simulation.backends.pytorch import PyTorchSimulator
from calphaebm.simulation.base import Simulator
from calphaebm.training.balancing import BalanceReport, recommend_lambdas
from calphaebm.training.losses.contrastive import contrastive_logistic_loss

# Training
from calphaebm.training.losses.dsm import dsm_cartesian_loss

__all__ = [
    # Version
    "__version__",
    # Models
    "TotalEnergy",
    "LocalEnergy",
    "RepulsionEnergy",
    "RepulsionEnergyFixed",
    "RepulsionEnergyLearnedRadius",
    "SecondaryStructureEnergy",
    "PackingEnergy",
    # Geometry
    "bond_lengths",
    "bond_angles",
    "torsions",
    "dihedral",
    # Data
    "make_extended_chain",
    "random_sequence",
    # Simulation
    "PyTorchSimulator",
    "Simulator",
    # Training
    "dsm_cartesian_loss",
    "contrastive_logistic_loss",
    "recommend_lambdas",
    "BalanceReport",
    # Evaluation
    "rmsd_kabsch",
    "q_hard",
    "q_smooth",
    "contact_count",
    "radius_of_gyration",
    "EvaluationReport",
    "TrajectoryEvaluator",
]
