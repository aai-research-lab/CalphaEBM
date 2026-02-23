# src/calphaebm/evaluation/__init__.py

"""Evaluation metrics for trajectory analysis."""

from calphaebm.evaluation.io.loaders import load_coords_from_pt, load_coords_from_xyz
from calphaebm.evaluation.io.writers import save_metrics_json, save_metrics_txt
from calphaebm.evaluation.metrics.clash import clash_probability, min_nonbonded
from calphaebm.evaluation.metrics.contacts import (
    contact_count,
    native_contact_set,
    q_hard,
    q_smooth,
)
from calphaebm.evaluation.metrics.rdf import rdf_counts, rdf_normalized
from calphaebm.evaluation.metrics.rg import radius_of_gyration
from calphaebm.evaluation.metrics.rmsd import kabsch_rotate, rmsd_kabsch
from calphaebm.evaluation.plotting import (
    plot_all,
    plot_min_distance,
    plot_q,
    plot_rdf,
    plot_rg,
    plot_rmsd,
)
from calphaebm.evaluation.reporting import EvaluationReport, TrajectoryEvaluator

__all__ = [
    # Metrics
    "rmsd_kabsch",
    "kabsch_rotate",
    "native_contact_set",
    "q_hard",
    "q_smooth",
    "contact_count",
    "rdf_counts",
    "rdf_normalized",
    "radius_of_gyration",
    "min_nonbonded",
    "clash_probability",
    # I/O
    "load_coords_from_pt",
    "load_coords_from_xyz",
    "save_metrics_json",
    "save_metrics_txt",
    # Reporting
    "EvaluationReport",
    "TrajectoryEvaluator",
    # Plotting
    "plot_rg",
    "plot_rmsd",
    "plot_q",
    "plot_min_distance",
    "plot_rdf",
    "plot_all",
]
