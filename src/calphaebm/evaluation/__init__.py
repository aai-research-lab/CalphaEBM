# src/calphaebm/evaluation/__init__.py

"""Evaluation metrics for trajectory analysis."""

from calphaebm.evaluation.metrics.rmsd import rmsd_kabsch, kabsch_rotate
from calphaebm.evaluation.metrics.contacts import (
    native_contact_set,
    q_hard,
    q_smooth,
    contact_count,
)
from calphaebm.evaluation.metrics.rdf import rdf_counts, rdf_normalized
from calphaebm.evaluation.metrics.rg import radius_of_gyration
from calphaebm.evaluation.metrics.clash import min_nonbonded, clash_probability
from calphaebm.evaluation.io.loaders import load_coords_from_pt, load_coords_from_xyz
from calphaebm.evaluation.io.writers import save_metrics_json, save_metrics_txt
from calphaebm.evaluation.reporting import EvaluationReport, TrajectoryEvaluator
from calphaebm.evaluation.plotting import (
    plot_rg,
    plot_rmsd,
    plot_q,
    plot_min_distance,
    plot_rdf,
    plot_all,
)

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