"""Plotting functions for evaluation results."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_rg(
    rg_series: np.ndarray,
    rg_ref: float,
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot radius of gyration vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rg_series))
    ax.plot(x, rg_series, "b-", linewidth=1.5, label="Rg(t)")
    ax.axhline(rg_ref, color="k", linestyle="--", linewidth=1.0, label="Reference")

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Rg (Å)", fontsize=12)
    ax.set_title("Radius of Gyration", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_delta_rg(
    delta_rg_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot ΔRg vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(delta_rg_series))
    ax.plot(x, delta_rg_series, "b-", linewidth=1.5)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("ΔRg (Å)", fontsize=12)
    ax.set_title("ΔRg vs Reference", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_rmsd(
    rmsd_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot RMSD vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rmsd_series))
    ax.plot(x, rmsd_series, "r-", linewidth=1.5)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title("Cα RMSD to Reference", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_q(
    q_series: np.ndarray,
    title: str = "Q (Native Contacts)",
    color: str = "g",
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot Q vs frame."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(q_series))
    ax.plot(x, q_series, f"{color}-", linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Q", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_q_comparison(
    q_hard_series: np.ndarray,
    q_smooth_series: np.ndarray,
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot Q_hard and Q_smooth together."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(q_hard_series))
    ax.plot(x, q_hard_series, "b-", linewidth=1.5, label="Q_hard")
    ax.plot(x, q_smooth_series, "r-", linewidth=1.5, label="Q_smooth")
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Q", fontsize=12)
    ax.set_title("Native Contact Fraction", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_min_distance(
    min_series: np.ndarray,
    median_series: np.ndarray,
    clash_threshold: float = 3.8,
    out_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot min and median nonbonded distances."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(min_series))
    ax.plot(x, min_series, "r-", linewidth=1.5, label="Min")
    ax.plot(x, median_series, "b-", linewidth=1.5, label="Median")
    ax.axhline(
        clash_threshold,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label=f"Clash ({clash_threshold:.1f} Å)",
    )

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Distance (Å)", fontsize=12)
    ax.set_title("Nonbonded Distances", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    return fig


def plot_rdf(
    centers: np.ndarray,
    counts: np.ndarray,
    norm: np.ndarray,
    out_dir: Optional[Path] = None,
    dpi: int = 200,
) -> Dict[str, plt.Figure]:
    """Plot RDF in various forms."""
    figs = {}

    # Raw counts
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(centers, counts, width=centers[1] - centers[0], alpha=0.7)
    ax1.set_xlabel("r (Å)", fontsize=12)
    ax1.set_ylabel("Counts", fontsize=12)
    ax1.set_title("RDF - Raw Counts", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["counts"] = fig1

    # Normalized
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(centers, norm, "b-", linewidth=2)
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("r (Å)", fontsize=12)
    ax2.set_ylabel("g(r)", fontsize=12)
    ax2.set_title("RDF - Normalized", fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["norm"] = fig2

    if out_dir:
        fig1.savefig(out_dir / "rdf_counts.png", dpi=dpi)
        fig2.savefig(out_dir / "rdf_norm.png", dpi=dpi)
        plt.close(fig1)
        plt.close(fig2)

    return figs


def plot_all(
    report: "EvaluationReport",
    out_dir: Path,
    dpi: int = 200,
) -> None:
    """Generate all plots for an evaluation report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rg plots
    plot_rg(report.rg_series, report.rg_ref, out_dir / "rg.png", dpi)
    plot_delta_rg(report.delta_rg_series, out_dir / "delta_rg.png", dpi)

    # RMSD
    plot_rmsd(report.rmsd_series, out_dir / "rmsd.png", dpi)

    # Q plots
    if report.q_hard_series.size > 0:
        plot_q(report.q_hard_series, "Q_hard", "b", out_dir / "q_hard.png", dpi)

    if report.q_smooth_series.size > 0:
        plot_q(report.q_smooth_series, "Q_smooth", "r", out_dir / "q_smooth.png", dpi)

    if report.q_hard_series.size > 0 and report.q_smooth_series.size > 0:
        plot_q_comparison(
            report.q_hard_series,
            report.q_smooth_series,
            out_dir / "q_comparison.png",
            dpi,
        )

    # Distance plots - fixed to use out_path, not out_dir
    plot_min_distance(
        report.min_distances_series,
        report.min_distances_series,  # Using same for min and median (temporary)
        clash_threshold=3.8,
        out_path=out_dir / "min_distance.png",
        dpi=dpi,
    )

    # RDF
    plot_rdf(report.rdf_centers, report.rdf_counts, report.rdf_norm, out_dir, dpi)
