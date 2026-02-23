"""High-level evaluation reporting."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from calphaebm.evaluation.io.loaders import load_coords_from_pt, load_coords_from_xyz, load_trajectory_from_dir
from calphaebm.evaluation.io.writers import save_metrics_json, save_metrics_txt
from calphaebm.evaluation.metrics.clash import batch_min_distances, clash_probability
from calphaebm.evaluation.metrics.contacts import contact_count, native_contact_set, q_hard, q_smooth
from calphaebm.evaluation.metrics.rdf import batch_rdf
from calphaebm.evaluation.metrics.rg import batch_rg, radius_of_gyration
from calphaebm.evaluation.metrics.rmsd import batch_rmsd


@dataclass
class EvaluationReport:
    """Comprehensive evaluation results for a trajectory."""

    # Basic info
    n_frames: int
    n_atoms: int
    reference_label: str

    # Metrics (scalars)
    rg_ref: float
    rg_mean: float
    rg_std: float
    delta_rg_mean: float
    delta_rg_std: float

    rmsd_mean: float
    rmsd_std: float

    q_hard_mean: Optional[float] = None
    q_hard_std: Optional[float] = None
    q_smooth_mean: Optional[float] = None
    q_smooth_std: Optional[float] = None

    contacts_mean: float = 0.0
    contacts_std: float = 0.0
    native_contacts: int = 0

    min_distance_median_mean: float = 0.0
    min_distance_absolute_min: float = 0.0
    clash_probability_all: float = 0.0
    clash_probability_post_burnin: float = 0.0

    # Time series
    rmsd_series: np.ndarray = field(default_factory=lambda: np.array([]))
    rg_series: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_rg_series: np.ndarray = field(default_factory=lambda: np.array([]))
    q_hard_series: np.ndarray = field(default_factory=lambda: np.array([]))
    q_smooth_series: np.ndarray = field(default_factory=lambda: np.array([]))
    contacts_series: np.ndarray = field(default_factory=lambda: np.array([]))
    min_distances_series: np.ndarray = field(default_factory=lambda: np.array([]))

    # RDF
    rdf_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    rdf_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    rdf_norm: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("CalphaEBM Evaluation Summary")
        lines.append("=" * 60)
        lines.append(f"Frames: {self.n_frames}")
        lines.append(f"Atoms: {self.n_atoms}")
        lines.append(f"Reference: {self.reference_label}")
        lines.append("")
        lines.append(f"Rg_ref: {self.rg_ref:.3f} A")
        lines.append(f"Rg_mean: {self.rg_mean:.3f} +/- {self.rg_std:.3f} A")
        lines.append(
            f"Delta Rg: {self.delta_rg_mean:.3f} +/- {self.delta_rg_std:.3f} A"
        )
        lines.append("")
        lines.append(f"RMSD_mean: {self.rmsd_mean:.3f} +/- {self.rmsd_std:.3f} A")
        lines.append("")
        if self.q_hard_mean is not None:
            lines.append(f"Q_hard: {self.q_hard_mean:.3f} +/- {self.q_hard_std:.3f}")
        if self.q_smooth_mean is not None:
            lines.append(
                f"Q_smooth: {self.q_smooth_mean:.3f} +/- {self.q_smooth_std:.3f}"
            )
        lines.append("")
        lines.append(f"Contacts: {self.contacts_mean:.1f} +/- {self.contacts_std:.1f}")
        lines.append(f"Native contacts: {self.native_contacts}")
        lines.append("")
        lines.append(
            f"Min distance (median mean): {self.min_distance_median_mean:.3f} A"
        )
        lines.append(
            f"Min distance (absolute min): {self.min_distance_absolute_min:.3f} A"
        )
        lines.append(f"Clash probability (all): {self.clash_probability_all:.4f}")
        lines.append(
            f"Clash probability (post-burnin): {self.clash_probability_post_burnin:.4f}"
        )
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(
        self, out_dir: Path, prefix: str = "eval", generate_plots: bool = True
    ) -> None:
        """Save all results to directory.

        Args:
            out_dir: Output directory.
            prefix: Prefix for output files.
            generate_plots: Whether to generate plots (requires plotting module).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Text summary
        save_metrics_txt({"summary": self.summary()}, out_dir / f"{prefix}_summary.txt")

        # JSON (scalars only)
        scalars = {
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "rg_ref": self.rg_ref,
            "rg_mean": self.rg_mean,
            "rg_std": self.rg_std,
            "delta_rg_mean": self.delta_rg_mean,
            "delta_rg_std": self.delta_rg_std,
            "rmsd_mean": self.rmsd_mean,
            "rmsd_std": self.rmsd_std,
            "q_hard_mean": self.q_hard_mean,
            "q_hard_std": self.q_hard_std,
            "q_smooth_mean": self.q_smooth_mean,
            "q_smooth_std": self.q_smooth_std,
            "contacts_mean": self.contacts_mean,
            "contacts_std": self.contacts_std,
            "native_contacts": self.native_contacts,
            "min_distance_median_mean": self.min_distance_median_mean,
            "min_distance_absolute_min": self.min_distance_absolute_min,
            "clash_probability_all": self.clash_probability_all,
            "clash_probability_post_burnin": self.clash_probability_post_burnin,
        }
        save_metrics_json(scalars, out_dir / f"{prefix}_scalars.json")

        # Time series (CSV)
        series = {
            "frame": np.arange(len(self.rmsd_series)),
            "rmsd": self.rmsd_series,
            "rg": self.rg_series,
            "delta_rg": self.delta_rg_series,
            "contacts": self.contacts_series,
            "min_distance": self.min_distances_series,
        }
        if self.q_hard_series.size > 0:
            series["q_hard"] = self.q_hard_series
        if self.q_smooth_series.size > 0:
            series["q_smooth"] = self.q_smooth_series

        import csv

        with open(out_dir / f"{prefix}_timeseries.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(series.keys())
            for i in range(len(self.rmsd_series)):
                writer.writerow([series[k][i] for k in series.keys()])

        # Generate plots if requested
        if generate_plots:
            try:
                from calphaebm.evaluation.plotting import plot_all

                plot_all(self, out_dir)
            except ImportError as e:
                print(f"Warning: Could not generate plots: {e}")


class TrajectoryEvaluator:
    """High-level trajectory evaluation."""

    def __init__(
        self,
        contact_cutoff: float = 8.0,
        exclude: int = 2,
        rdf_rmax: float = 20.0,
        rdf_dr: float = 0.25,
        q_smooth_beta: float = 5.0,
        q_smooth_lambda: float = 1.8,
        clash_threshold: float = 3.8,
    ):
        self.contact_cutoff = contact_cutoff
        self.exclude = exclude
        self.rdf_rmax = rdf_rmax
        self.rdf_dr = rdf_dr
        self.q_smooth_beta = q_smooth_beta
        self.q_smooth_lambda = q_smooth_lambda
        self.clash_threshold = clash_threshold

    def evaluate(
        self,
        trajectory: List[np.ndarray],
        reference: np.ndarray,
        reference_label: str = "reference",
        burnin: int = 0,
        compute_q_smooth: bool = True,
    ) -> EvaluationReport:
        """Evaluate trajectory against reference."""

        # Convert to array
        traj_array = np.array(trajectory)  # (n_frames, N, 3)
        n_frames, n_atoms = traj_array.shape[:2]

        # Reference native contacts
        native_i, native_j, native_d0 = native_contact_set(
            reference, cutoff=self.contact_cutoff, exclude=self.exclude
        )
        n_native = len(native_i)

        # Compute metrics for all frames
        rmsds = batch_rmsd(traj_array, reference)
        rgs = batch_rg(traj_array)
        delta_rgs = rgs - radius_of_gyration(reference)

        contacts = np.array(
            [
                contact_count(frame, self.contact_cutoff, self.exclude)
                for frame in traj_array
            ]
        )

        mins, medians = batch_min_distances(traj_array, self.exclude)

        # Q metrics
        q_hards = None
        q_smooths = None

        if n_native > 0:
            q_hards = np.array(
                [
                    q_hard(frame, native_i, native_j, self.contact_cutoff)
                    for frame in traj_array
                ]
            )

            if compute_q_smooth:
                q_smooths = np.array(
                    [
                        q_smooth(
                            frame,
                            native_i,
                            native_j,
                            native_d0,
                            self.q_smooth_beta,
                            self.q_smooth_lambda,
                        )
                        for frame in traj_array
                    ]
                )

        # RDF
        rdf_centers, rdf_counts, rdf_norm = batch_rdf(
            traj_array, self.rdf_rmax, self.rdf_dr, self.exclude
        )

        # Clash probabilities
        p_all, p_post = clash_probability(
            traj_array, self.clash_threshold, self.exclude, burnin
        )

        # Burnin indices
        if burnin > 0 and burnin < n_frames:
            post_idx = slice(burnin, None)
        else:
            post_idx = slice(None)

        # Create report
        report = EvaluationReport(
            n_frames=n_frames,
            n_atoms=n_atoms,
            reference_label=reference_label,
            rg_ref=radius_of_gyration(reference),
            rg_mean=float(rgs.mean()),
            rg_std=float(rgs.std()),
            delta_rg_mean=float(delta_rgs.mean()),
            delta_rg_std=float(delta_rgs.std()),
            rmsd_mean=float(rmsds.mean()),
            rmsd_std=float(rmsds.std()),
            q_hard_mean=float(q_hards.mean()) if q_hards is not None else None,
            q_hard_std=float(q_hards.std()) if q_hards is not None else None,
            q_smooth_mean=float(q_smooths.mean()) if q_smooths is not None else None,
            q_smooth_std=float(q_smooths.std()) if q_smooths is not None else None,
            contacts_mean=float(contacts.mean()),
            contacts_std=float(contacts.std()),
            native_contacts=n_native,
            min_distance_median_mean=float(medians[post_idx].mean()),
            min_distance_absolute_min=float(mins.min()),
            clash_probability_all=p_all,
            clash_probability_post_burnin=p_post,
            rmsd_series=rmsds,
            rg_series=rgs,
            delta_rg_series=delta_rgs,
            q_hard_series=q_hards if q_hards is not None else np.array([]),
            q_smooth_series=q_smooths if q_smooths is not None else np.array([]),
            contacts_series=contacts,
            min_distances_series=mins,
            rdf_centers=rdf_centers,
            rdf_counts=rdf_counts,
            rdf_norm=rdf_norm,
        )

        return report

    def evaluate_from_dir(
        self,
        traj_dir: str,
        ref_path: Optional[str] = None,
        burnin: int = 0,
    ) -> EvaluationReport:
        """Load trajectory from directory and evaluate."""

        # Load trajectory
        frames = load_trajectory_from_dir(traj_dir)

        # Load reference
        if ref_path is None:
            ref_path = os.path.join(traj_dir, "coords.pt")

        if ref_path.endswith(".pt"):
            reference = load_coords_from_pt(ref_path)
        elif ref_path.endswith(".xyz"):
            reference = load_coords_from_xyz(ref_path)
        elif ref_path.endswith(".pdb"):
            # Load PDB file
            import mdtraj as md

            traj = md.load_pdb(ref_path)
            reference = traj.xyz[0]  # Take first frame
        else:
            raise ValueError(f"Unknown reference format: {ref_path}")

        return self.evaluate(
            frames,
            reference,
            reference_label=os.path.basename(ref_path),
            burnin=burnin,
        )
