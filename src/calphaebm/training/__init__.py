# src/calphaebm/training/__init__.py

"""Training utilities for phased energy model training."""

from calphaebm.training.balancing import BalanceReport, recommend_lambdas, term_forces
from calphaebm.training.freeze import freeze_module, set_requires_grad, unfreeze_module
from calphaebm.training.losses.contrastive import contrastive_logistic_loss
from calphaebm.training.losses.dsm import dsm_cartesian_loss
from calphaebm.training.phased import PhasedTrainer

__all__ = [
    # Losses
    "dsm_cartesian_loss",
    "contrastive_logistic_loss",
    # Phased training
    "PhasedTrainer",
    # Freezing utilities
    "freeze_module",
    "unfreeze_module",
    "set_requires_grad",
    # Lambda balancing
    "recommend_lambdas",
    "BalanceReport",
    "term_forces",
]
