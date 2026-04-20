"""Core training utilities for phased training."""

from calphaebm.training.core.state import TrainingState, ValidationMetrics
from calphaebm.training.core.trainer import BaseTrainer
from calphaebm.training.core.convergence import ConvergenceMonitor, ConvergenceCriteria
from calphaebm.training.core.config import PhaseConfig
from calphaebm.training.core.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from calphaebm.training.core.freeze import freeze_module, unfreeze_module, set_requires_grad
from calphaebm.training.core.schedules import apply_gate_schedule, get_lr

__all__ = [
    'TrainingState',
    'ValidationMetrics',
    'BaseTrainer',
    'ConvergenceMonitor',
    'ConvergenceCriteria',
    'PhaseConfig',
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'freeze_module',
    'unfreeze_module',
    'set_requires_grad',
    'apply_gate_schedule',
    'get_lr',
]